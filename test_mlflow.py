import mlflow 
mlflow.set_experiment('carvana-unet') 
with mlflow.start_run(): 
    mlflow.log_param('test', 'hello') 
    mlflow.log_metric('test_metric', 0.5, step=0) 
    print('MLflow logging works correctly') 
