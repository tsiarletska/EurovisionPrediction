# EurovisionPrediction


## Initial data: 
**data.csv** : file with songs data of Eurovision form 2000 to 2023. Contains only songs qualifies for the final.  Lyrics are not cleaned.
Contains columns: Year,Place,Points,No.,Country,Performer,Song,Lyrics. Lyrics are cleaned. 
**test_2024.csv**: file with songs form 2024 final
Contains columns: Year,Lyrics,Country,Performer,Song,Place,Topic

## Models: 
**pred3.py** - the NN model, which predicts the placement based on cleared lyrics (noted on the songs parts and stop worlds in english)
**pred4.py** - the NN model,, whicj predicts the placement based on teh country, cleared lyrics, year, languages, topics
**pred5.py** - similar to pred4.py, but loops throught the models' parameters and saves the results of the metrics

## Prediction results: 
**pred3.py** - generates file "eurovision_predictions.csv" with Country,Song,Performer,Actual_Place,Predicted_Place,Prediction_Error
**pred4.py** - generates file "results_pred4.csv" with Song,Performer,Year,Predicted_Place,Actual_Place\\
**pred5.py** - generated file "hyperparameter_results.csv" with activation,lr,batch_size,epochs,MAE,MSE,Spearman


## Conclusions
MAE are for all prediction is around 5, which is pretty low considering that we have 25 songs in the test_2024.csv
Checking the output file for the pred4.py: all song prediction are around 11th place. 
I'll better use the **pred3.py**, even thought the metrics are also average for this prediction, but to the human eyes, the prediction are close to the reality (ofc not all of them.)

## Additional file
**"hyperparameter_results_all.csv"** - file, which collects results of the models' performaces (has the same columns as an "hyperparameter_results.csv").
Paying with different activation functions, batch_size, number of epochs, learning rates and metrics of assessments - MAE, MSE and Spearman.
