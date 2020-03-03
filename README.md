# Analyzing Weld Composite Material and Mechanial properties using statistical / machine learning approaches

## Webscraping



### The data for this projects was obtained via webscraping welding materials data from the following sites

- [Certilas Welding Products Dataset](https://certilas.nl/en/catalogue)
- [Lincoln Electric Welding Products Dataset](https://www.lincolnelectric.com/en-us/consumables/Pages/consumables.aspx)
- [Citrination Weld Composites Dataset](https://citrination.com/datasets/114194/show_search?searchMatchOption=fuzzyMatch)

This scraping was done through the combination of webscraping and data preprocessing using BeautifulSoup and Pandas. See [my webscraping code to see exactly how I did that (sorry about the messiness)](https://github.com/PrasannS/WeldingAnalysis/blob/master/notebooks/webscraper.ipynb)

To process the Lincoln Dataset I had to do a bit of custom data extraction with textract, tabula, PDFPlumber, and a couple of other tools.

## Statistical Analysis



### Again, this took a lot of work with pandas, scipy and statistical libraries. Lots of SKLearn.

All of my statistical analysis that I performed on the dataset I assembled can be found [here](https://github.com/PrasannS/WeldingAnalysis/blob/master/notebooks/stat_analysis.ipynb)

I spent a lot of time examining property counts, looking for variance, covariance. T_test, z_test, chi square, and getting p-values for all of these tests was necessary to gain an intimate understanding of the data. I went on to do some [residual plotting](https://github.com/PrasannS/WeldingAnalysis/blob/master/notebooks/regression_analysis.ipynb) which I started off early before doing any regression to analyze homo/heteroskedasticity, and I ended up doing a good bit of structural data manipulation to try and create more distributed residuals (I made some progress fortunately). 

## Regression analysis and Gaussian Process Analysis


### Approach at creating models

Since multivariate models seemed a bit too complex for this usecase, I stuck to training 1 target value at a time, going though all of the Material and Mechanical Properties that could be found in my dataset. I optimized each model individually on each target parameter to create persoanalized models for each individual case. These were optimized individually, and finally stored in benchmark tables / model pickles (TODO). I used RMSE as my evaluation metric.  

### This entailed a wide variety of basically all the models that seemed like good fits for the data

I started off with simple linear and logistic regression models. I wrote up some code in my [early methods](https://github.com/PrasannS/WeldingAnalysis/blob/master/notebooks/regression_analysis.ipynb) to test individual models with set hyperparameters quickly before moving on to more comprehensive approaches. I was surprised to find that a large majority of the models performed relatively similarly. Looking at the data, a decision tree structure stood out as a better fit, but RandomForest models performed a bit better than I expected for many of the properties. 

### GP with GPy

I [checked out](https://github.com/PrasannS/WeldingAnalysis/blob/master/notebooks/gaussian_analysis.ipynb) some Gaussian Process Regression out to compare it to some of my other models. I did a lot of experimentation with models with these techniques, but it in general didn't go as well as I would've hoped. Hyperparameter tuning was interesting, as there were basically endless possibilities, and limited runtime since the models took so long. The RBF kernel performed nicely on it's own, but even basic transformation such as multiplication with linear kernels led to some improvements, so the jury very well may be still out this approach. At the end of the day, the best models only seemed to match up with my less performing models. Some reasons were probably: 
- Lack of a proper methodology to hyperparameter tund and mess around with the model
- Lack of experience
- Huge runtimes
- Most of the data probably wasn't a good fit because of the abnormality / skew nature of much of my data

I would definetely like to look into it more if possible, it was a great introduction though. 

### Model exploration continued...

Once I handled many of the more basic models provided by Scikit Learn (RandomForest Regressor, DecisionTree, LARS, etc.), I moved on to some other models known to perform, outside of the given libraries. XGBoost (trusty as always), Light GBMs, and CatBoost were the main ones that I tried out. XGBoost by far performed the best, and in general beat almost every other algorithm I've tested, after proper hyperparameter tuning. This was the stage where I broke out HyperOpt and started extracting the full potential of my models, but without a doubt there is still plenty of work to be done. (TODO) A VERY PESKY BUG ruined my week, but was the primary hitch in this process. In general, while very new to me, this entire stage was relatively methodical, if nothing else. I would not call my model search exhaustive, but comprehensive (at least considering how new a lot of this work is to me).


