## Brief Summary

For our current project, our goal is to perform a simple survival analysis on the Bone Marrow Transplant Children dataset. We will divide our current population into different clusters to see the potential impact of certain features on the survival probability of each group.

## EDA and Data Preprocessing

We first start our EDA by downloading the dataset which has 187 patients, 85 of them deceased while 102 of them right censored, as well as 36 features. We first computed the correlation between features and saw different highly correlated groups of variables.Additionally,Some of these features could be obtained from others, so we dropped them to reduce collinearity, plus we moved survival_time from the features to the targets.

Next up, we removed some variables to avoid leakage, as these ones are only known after transplant, as our aim is to predict survival time at the moment of the transplant. We then imputed missing values to, \texttt{RecipientCMV},  \texttt{CD3dkgx10d8}, \texttt{DonorCMV}, \texttt{Rbodymass}, \texttt{RecipientRh}, \texttt{Allele}, \texttt{Antigen}, and \texttt{Recipient ABO} . We used mode, median and an iterative imputer for categorical, numerical, and high percentage of missing features.

Now that the data was clean we performed Pearson and Spearman correlations of each feature with \texttt{survival\_time} and \texttt{survival\_status}. Since \texttt{Disease} and \texttt{DonorABO} were weak on both targets, we dropped them. We then evaluated feature importance by using a Random Survival Forest. We kept the features with positive importance giving us 8 features remaining. Finally we used PCA to visualize the final dataset, where the first two principal components explain around 52\% of the variance.

## Clustering

Now that we had our dataset ready we move to clustering, where we try to group patients depending on their features in different subgroups. We first ran K-Means, but the results were not really convincing after some analysis with PCA, thus we use Spectral Clustering. The metrics were very similar for every K (silhouette around 0.2), so the data does not show strongly separated groups, and we chose K=2 as the simplest and most interpretable option, meaning categorizing the patients into two subgroups.

After this, we analyzed the resulting clusters. For the first one we obtained 146 patients, while 41 for the second, with a death rate of 0.42 and 0.59 respectively. The rest of the features were similar between the two subgroups except for CD34+ and CD3+ dose per kg as well as the stem cell source. We can conclude that the model clusters by patients with peripheral blood transplants (99\%) with high CD34+ and CD3+ doses against patients with bone marrow transplants (100\%) with much lower CD34+ and CD3+ doses.

## Survival Analysis

We first use Kaplan-Meier to create survival curves on both subgroups. After the first year the first subgroup of patients has around 17 percentage points more survival probability (68.5\% vs 51.2\%), and the log-rank test gives $p = 0.03$, meaning the difference between both groups is statistically significant at the 5\% level. We then compare the survival function from Kaplan-Meier against the one with Cox using the cluster as a covariate. Cox gives us a hazard ratio of 1.68 (95\% CI 1.04--2.69) with $p = 0.03$, meaning the patients in the second subgroup have a 68\% higher risk of death at any time.

Comparing both methods, Kaplan-Meier is non-parametric and only compares groups. On the other hand, Cox is semi-parametric, since it assumes a constant hazard ratio, summarizes the difference in a single number, can adjust for several covariates and provides feature importance. The proportional hazards assumption holds (Schoenfeld test $p = 0.24$), which is why both curves match. When adding recipient age, risk group and disease group to the model, the cluster effect remains (HR 1.69, $p = 0.03$), so these variables do not explain the difference. Regarding feature importance, a Cox model on the standardized features shows no feature significant at the 5\% level, the most relevant ones being body mass (HR 1.53 per SD, $p = 0.10$), risk group (HR 1.20, $p = 0.095$) and CD3+ dose (HR 0.78, $p = 0.15$), with a C-index of 0.64.


## Conclusion

To conclude, we see that patients do not form well separated subgroups (silhouette of 0.22), as Spectral Clustering groups them by stem cell source, peripheral blood with high CD34+ and CD3+ doses against bone marrow with low doses. The second group has a worse survival, with HR=1.68, which is partly aligned with the hypothesis of the original study, claiming that higher cell doses improves survival. Though once all features are considered the CD34+ shows no effect and the CD3+ is the closest to being relevant. We think that the main limitation of the study is the small dataset and the filter of features depending on the survival on the EDA, which may make the difference between the clusters bigger than it actually is.