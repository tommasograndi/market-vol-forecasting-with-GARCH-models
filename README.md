# market-vol-forecasting-with-GARCH-models

Welcome everyone, this GitHub repository host a project focused on volatility forecasting and modeling, with a specific emphasis on the FTSE MIB 100 index. The project explores key anomalies in financial returns, such as the leverage effect, volatility clustering and leptokurticity. By leveraging GARCH and GARCH-derived models, including EGARCH, TGARCH, and GJR GARCH, the goal is to identify the most suitable model for forecasting the volatility of the FTSE MIB 100 index.

The structure of the project is the following. 

The notebook `notebook_GARCH` contain most of the project and models made in Python. It first starts by discussing and showing the anomalies in financial returns and volatility in order to motivate the use of GARCH models. Successively the notebook showcase the use of GARCH, EGARCH and GJR-GARCH models to forecast the volatility and compared those models through model evaluation and error measures. 

The R markdown `TGARCH_R.rmd` contained in the folder `TGARCH_R` instead contains discuss the TGARCH model and compare it to all the other models contained in the Python notebook.

Conclusions are then drawn in the Python notebook and the best model is choose.

