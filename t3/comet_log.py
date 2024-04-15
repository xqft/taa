def save_gs_results(cmt_exp, grid_search):
    '''
    Entrada:
        cmt_exp: experimento comet
        cv_results: scikit-learn grid search results
    '''
    
    index = grid_search.best_index_
    results = grid_search.cv_results_
    if grid_search.cv == None:
        cv = 5
    else: 
        cv = grid_search.cv
        
    ### Resultados para cada fold ###
    for k in range(cv):
        metrics_step = {
            'Val RMSLE': -results[f'split{k}_test_score'][index],
            'Train RMSLE': -results[f'split{k}_train_score'][index],
        }
        cmt_exp.log_metrics(metrics_step,step=k)

    return