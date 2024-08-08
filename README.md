# Objective Fairness Index (OFI)

Here, we present empirical results and code to verify our theoretical results (example confusion matrices and 
the standard deviation of marginal benefit).

## To Run

Please create a virtual environment with Python 3.11, navigate to the top directory (contains requirements.txt), 
then run:
```python
pip install -r requirements.txt
```

Then, each file has a main code block that is run only when the file is executed to generate the results.
This code block is found at the bottom of the file, under ```if __name__ == '__main__':```.

## Empirical Results

Relevant files: compas_case_study.py and folktables_case_study.py

Results: results/

We present four case studies from two well-regarded baselines: COMPAS (risk of recidivism) and Folktable’s Adult Employment (is person employed) datasets. 

The images in results/ are heatmaps for treatment equality (TE), predictive parity (PP), DI, and OFI scores, compared with all race pairs.

In compas_ethnicity.png, we see that Objective Fairness Index affirms preliminary findings of DI. This is an important revelation, as DI is susceptible to arguments and cases of higher prediction rates being correct. In return, some argue that higher rates would be due to systematic bias. However, Objective Fairness Index sidesteps the need to prove incorrect ground labels and shows that even if one assumes correct ground labels, there’s algorithmic bias against certain ethnicities. We believe this revelation will encourage more people to take corrective measures.

In the Folktable’s cases, we investigate Georgia’s census database from 2014 to 2017 (a total of 393,236 observations) and predict if an individual is employed. Using the random forest classifier (RF), folk_rf.png reveals that Objective Fairness Index corroborates with DI in this case study, as it was in COMPAS. However, in the Naive Bayes (NB) case study with Folktables, we find that Objective Fairness Index strongly disagrees with DI when context necessitates it.

In the NB case study, Objective Fairness Index strongly disagrees on DI’s findings regarding Pacific Islanders. DI indicates that Pacific Islanders have bias for them compared to 7/8 other races while Objective Fairness Index shows that, with context, Pacific Islanders have bias for them over only 2/8 other races. Among the most prominent distinction occurs when comparing with Whites: DI suggests large bias for Pacific Islanders while Objective Fairness Index shows a slight bias against them (folk_nb.png).

We further investigate Objective Fairness Index’s nuances by drawing 60 random samples to mimic a smaller dataset. In folk_rf_few.png, we see TE and DI’s problematic undefined cases. Furthermore, the “Two or more races” holds a constant 0 in DI, while Objective Fairness Index gives more information. OFI's nuance still shows substantial bias against the mixed-race class but indicates that they are more susceptible to bias when compared with whites as opposed to blacks.

Our practical use cases show that Objective Fairness Index gives unique insights to understand the manifested bias. 

Notes: due to DI’s asymmetry, the heatmap's color map's gradient centers toward blue instead of gray like TE and OFI do. Furthermore, TE’s gradient is flipped as this definition has larger values indicate bias toward Race i, unlike the others. Finally, since we assume the positive label is preferrable, we flip the COMPAS labels so the positive is “not a recidivist”. 
## Hypothetical Confusion Matrices

We verify our metric results in binary_confusion_matrix.py.

## Proofs on Marginal Benefit's (B) Standard Deviation

We’ve verified our proofs in the appendix with a computer algebra system in behavior.py. 