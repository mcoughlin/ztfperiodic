""" PENQUINS - Processing ENormous Queries of kowalski Users INStantaneously """

__all__ = ['Kowalski', '__version__']


from copy import deepcopy
from bson.json_util import loads
from multiprocessing.pool import ThreadPool
from netrc import netrc
import os
import requests
from requests.adapters import HTTPAdapter, DEFAULT_POOLBLOCK, DEFAULT_POOLSIZE
from requests.packages.urllib3.util.retry import Retry
import secrets
import string
import traceback
from tqdm.auto import tqdm
from typing import Union


__version__ = '2.0.1'


Num = Union[int, float]
QueryPart = Union['task', 'result']
Method = Union['get', 'post', 'put', 'patch', 'delete']


DEFAULT_TIMEOUT = 5  # seconds
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = DEFAULT_TIMEOUT
        return super().send(request, **kwargs)


class Kowalski:
    """
        Class to communicate with a Kowalski instance
    """

    def __init__(
            self, username=None, password=None, token=None,
            protocol: str = 'https', host: str = 'kowalski.caltech.edu', port: int = 443,
            verbose: bool = False,
            **kwargs
    ):
        """
            username, password, token, protocol, host, port:
                Kowalski instance access credentials and address
                If password is omitted, then look up default credentials from
                the ~/.netrc file.
            timeout, pool_connections, pool_maxsize, max_retries, backoff_factor, pool_block:
                control requests.Session connection pool settings
            verbose:
                "Status, Kowalski!"
        """

        if (username is None) and (password is None) and (token is None):
            raise ValueError("Missing credentials: provide either username and password or token")

        if (username is not None) and (password is None):
            netrc_auth = netrc().authenticators(host)
            if netrc_auth:
                username, _, password = netrc_auth

        # Status, Kowalski!
        self.v = verbose

        self.protocol = protocol

        self.host = host
        self.port = port

        self.base_url = f'{self.protocol}://{self.host}:{self.port}'

        self.session = requests.Session()

        # Prevent Requests from attempting to do HTTP basic auth using a
        # matching username and password from the user's ~/.netrc file,
        # because kowalski will reject all HTTP basic auth attempts.
        self.session.trust_env = False

        # mount session adapters
        timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)
        pool_connections = kwargs.get("pool_connections", DEFAULT_POOLSIZE)
        pool_maxsize = kwargs.get("pool_maxsize", DEFAULT_POOLSIZE)
        max_retries = kwargs.get("max_retries", DEFAULT_RETRIES)
        backoff_factor = kwargs.get("backoff_factor", DEFAULT_BACKOFF_FACTOR)
        pool_block = kwargs.get("pool_block", DEFAULT_POOLBLOCK)

        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[405, 429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "PUT", "POST", "PATCH"]
        )
        adapter = TimeoutHTTPAdapter(
            timeout=timeout,
            max_retries=retries,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            pool_block=pool_block
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # set up authentication headers
        if token is None:
            self.username = username
            self.password = password
            self.token = self.authenticate()
        else:
            self.token = token

        self.headers = {'Authorization': self.token}

        self.methods = {
            'get': self.session.get,
            'post': self.session.post,
            'put': self.session.put,
            'patch': self.session.patch,
            'delete': self.session.delete
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # run shut down procedure
        self.close()
        return False

    def close(self):
        """Shutdown session gracefully
        :return:
        """
        try:
            self.session.close()
            return True
        except Exception as e:
            if self.v:
                print(e)
            return False

    def authenticate(self):
        """Authenticate user, return access token
        :return:
        """

        # post username and password, get access token
        auth = self.session.post(
            f'{self.base_url}/api/auth',
            json={"username": self.username, "password": self.password, "penquins.__version__": __version__}
        )

        if auth.status_code == requests.codes.ok:
            if self.v:
                print(auth.json())

            if 'token' not in auth.json():
                print('Authentication failed')
                raise Exception(auth.json().get('message', 'Authentication failed'))

            access_token = auth.json().get('token')

            if self.v:
                print('Successfully authenticated')

            return access_token

        raise Exception('Authentication failed')

    def api(self, data: dict, endpoint: str = None, method: Method = None):

        if endpoint is None:
            raise ValueError('Endpoint not specified')
        if method not in ['get', 'post', 'put', 'patch', 'delete']:
            raise ValueError(f'Unsupported method: {method}')

        if method.lower() != 'get':
            resp = self.methods[method.lower()](
                os.path.join(self.base_url, endpoint),
                json=data,
                headers=self.headers,
            )
        else:
            resp = self.methods[method.lower()](
                os.path.join(self.base_url, endpoint),
                params=data,
                headers=self.headers,
            )

        if resp.status_code == requests.codes.ok:
            return loads(resp.text)

        raise Exception('API call failed')

    def batch_query(self, queries, n_treads: int = 4):
        n_treads = min(len(queries), n_treads)

        with ThreadPool(processes=n_treads) as pool:
            if self.v:
                return list(tqdm(pool.imap(self.query, queries), total=len(queries)))
            return list(pool.imap(self.query, queries))

    def query(self, query):

        _query = deepcopy(query)

        # by default, all queries are not registered in the db and the task/results are stored on disk as json files
        # giving a significant execution speed up. this behaviour can be overridden.
        save = _query.get('kwargs', dict()).get('save', False)

        if save:
            if 'kwargs' not in _query:
                _query['kwargs'] = dict()
            if '_id' not in _query['kwargs']:
                # generate a unique hash id and store it in query if saving query in db on Kowalski is requested
                _id = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(32))

                _query['kwargs']['_id'] = _id

        resp = self.session.post(
            os.path.join(self.base_url, 'api/queries'),
            json=_query,
            headers=self.headers
        )

        return loads(resp.text)

    def ping(self) -> bool:
        """
            Ping Kowalski
        :return: True if connection ok, False otherwise
        """
        try:
            resp = self.session.get(
                os.path.join(self.base_url, ''),
                headers=self.headers,
            )

            if resp.status_code == requests.codes.ok:
                return True
            return False

        except Exception as _e:
            _err = traceback.format_exc()
            if self.v:
                print(_e)
                print(_err)
            return False
