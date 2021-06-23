rsync -av /home/michael.coughlin/ZTF/output_quadrants_Primary_DR3_HC/catalog/ECE_ELS_EAOV mcoughlin@schoty.caltech.edu:/gdata/Data/PeriodSearches/DR3/HCsmall/


rsync -av /home/michael.coughlin/ZTF/output_phenomenological_ids_DR2/catalog/compare/slices_merged mcoughlin@schoty.caltech.edu:/gdata/Data/Features/v1/dnn-ids-DR2

# LLO
rsync -av michael.coughlin@ldas-pcdev2.ligo-la.caltech.edu:/home/michael.coughlin/ZTF/output_quadrants_Primary_DR3/catalog .

#LHO 
rsync -av michael.coughlin@ldas-pcdev2.ligo-wa.caltech.edu:/home/michael.coughlin/ZTF/output_quadrants_Primary_DR3/catalog .

# IUCCA
rsync -av michael.coughlin@ldas-pcdev1.gw.iucaa.in:/home/michael.coughlin/ZTF/output_quadrants_EAOV_Primary_DR3/catalog .


rsync -av /home/michael.coughlin/ZTF/output_quadrants_ECE_20Fields_DR3/catalog/ECE/ mcoughlin@schoty.caltech.edu:/gdata/Data/PeriodSearches/v8/output_quadrants_ECE_20Fields_DR3/

scp *.h5 mcoughlin@schoty.caltech.edu:/gdata/Data/Features/v1/xgboost-ids-DR2/xgboost-merged

rsync -av /home/michael.coughlin/ZTF/output_quadrants_ECE_20Fields_DR3/catalog/ECE/ mcoughlin@schoty.caltech.edu:/gdata/Data/PeriodSearches/v8/output_quadrants_ECE_20Fields_DR3/ 

rsync -av  /home/michael.coughlin/ZTF/output_features_20Fields_ids_DR2/catalog/xgboost/ mcoughlin@schoty.caltech.edu:/gdata/Data/Features/v1/xgboost-20Fields/

rsync -av mcoughlin@supernova.caltech.edu:/media/Data/mcoughlin/Matchfiles ../../

