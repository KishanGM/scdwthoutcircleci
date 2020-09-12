from sklearn.pipeline import Pipeline

import preprocessors as pp


ME_VARS = [  'late_delivery'
                 ,'customer_country','customer_segment'
                 ,'samecountry_source_dest'
                 ,'ordered_on_weekends'
                 ,  'market'
                 , 'shipping_mode'
                 ,'order_dt_month','store_country','order_country','order_region'
                 ,'order_dt_weekday','cat_lowvol_lowrisk','cat_lowvol_highrisk', 'cat_highvol_lowrisk',
                'lowvol_lowrisk', 'highvol_lowrisk',
                'lowvol_highrisk', 'highvol_highrisk', 'order_country_logistics_performance_index'

]

#PIPELINE_NAME = 'lasso_regression'
#('Mean Encoding', DFMeanEncoding(mean_enc_cols,DFTrain[[target_col]]))
price_pipe = Pipeline(
    [
        ('categorical_imputer',
         pp.CategoricalImputer(variables=CATEGORICAL_VARS)),
    ])
