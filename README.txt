Classifying Animal Shelter Outcomes - Group Project


Problem Statement

Overpopulation of animals in shelters is a problem plaguing cities all over the US. This raises the ethical dilemma of euthanizing healthy animals when a shelter reaches capacity. To increase Austin's citywide life-saving rate, the Austin City Council implemented the 2010 No Kill Implementation Plan which includes components such as large-scale volunteering, foster care programs, and a partnership with Emancipet to provide affordable spay and neuter surgeries at a high volume. The goal of the plan is to help the city maintain a lifesaving rate of over 90 percent.

Predicting whether an animal will be euthanized or adopted based on past data can help shelters allocate and prepare the resources they will need such as staffing, space availability, and food to get as many animals adopted as possible and maintain Austin’s status as the nation’s largest No Kill city. A predictive model could help shelters decide which animals to put into foster care versus which are more likely to be adopted and should be made available sooner. In addition to the moral benefit of giving more animals permanent homes, a predictive model could save shelters money that would otherwise be spent on euthanasia methods and can instead go toward improving conditions of the shelter or hiring more staff. With an efficient and accurate classifier, animal shelters could lower their euthanasia rates and find permanent homes for more animals.


Data

Our original dataset contained 111,649 rows (not including headers) and 12 columns, although we ended up dropping more than half of these records as described in the next section. Our dataset consisted of animals who were under the care of the Austin Animal Center, their features, and their outcomes at the shelter (such as adopted, euthanised, transferred, etc.). This dataset contained data about outcomes from 2013 to 2019.

Specifically, each row in our original dataset is an animal at the Austin Animal Center with the following attributes as columns:

Animal ID - The ID given to the animal at the shelter
Animal name - The name given to the animal at the shelter
MonthYear - The date that the animal's outcome occurred
DateTime - A duplicate of MonthYear
Date of Birth - The date of birth of the animal
Outcome Type - The eventual outcome of the animal, such as adoption 
Outcome Subtype - Attributes of the animal such as rabies, aggressive, suffering, etc.
Animal Type - The type of animal, such as dog
Sex Upon Outcome - The sex of the animal upon the outcome and if they were spayed/neutered, such as intact male
Breed - The breed of the animal, such as terrier
Color - The color of the animal, such as black
Age Upon Outcome - The age of the animal upon the outcome, calculated using MonthYear minus Date of Birth 


Initial Data Cleaning

We started this project by cleaning the data directly in Microsoft Excel.

We removed the ID column and the Name column as they are not necessary to predict outcome (ID and name are arbitrarily given to the animals). 
We removed the original Age Upon Outcome column and replaced it with a new Age Upon Outcome column calculated by subtracting the Date of Birth from MonthYear, resulting in more precise ages in floating-point number format.
We removed the DateTime column because it was a duplicate of the MonthYear column.
We created four binary columns — TNR (trap-neuter-release), Rabies, Suffering, and Aggressive — from the Outcome Subtype column. We then dropped the original Outcome Subtype column that held strings. This made it possible to include these attributes when predicting with models that only use numerical values.

We then decided to eliminate, using python, all animals that were not available for adoption, leaving us with only dogs and cats. We also eliminated records with any outcome besides Adoption, Euthanasia, Transfer, or Return to Owner due to the small number of records that had these outcomes as well as the lack of information regarding their meaning (e.g. “Missing”). Lastly, we dropped records with null outcome types. For records with null Sex Upon Outcome values, we filled in the missing information with the most common Sex Upon Outcome value for each record’s Animal Type (e.g. “Neutered male”).  


Feature Engineering

Next, we feature engineered the Color attribute due to the large number of unique colors. We created two new columns: a binary Multicolor column to indicate if an animal is multicolored and a CombinedColor column with strings as values. Because colors appeared in ordered pairs such as “White/Black” as well as “Black/White,” we assumed the first color was the animal’s primary color. We then created the CombinedColor column by condensing similar main colors into groups (such as condensing all kinds of tabby coat colors into just “Tabby”). With 10 unique CombinedColor values, we eliminated the original Color column.

For the Breed attribute, we created a binary Pitbull column which indicates if a dog is part pitbull. We distinguished this breed due to the common stereotype of aggression surrounding pitbulls, which could have an effect on the outcomes for dogs with pitbull features. We condensed our breeds for cats and dogs down to seven unique breed groups for dogs, three unique breed groups for cats, and one shared breed between them, “Mutt.” We decided on these breed groups, including “Mutt,” using information from the American Kennel Club and Purina.

We then feature engineered with the MonthYear attribute and the Sex Upon Outcome attribute. Using the MonthYear attribute, we created a Season column based on the month in which an animal’s outcome occurred to account for patterns in adoptions during certain times of the year. For Sex Upon Outcome, we split up the two traits included in the original values and made two new columns: Spayed/Neutered and Gender. This split enabled the classifiers to evaluate these attributes distinctly when classifying records.
Additional Cleaning and Exploration

At this point, we noticed that some of the MonthYear values for some animals were at a date before the animal’s date of birth. Assuming this to be human error and because there were very few of these errors, we dropped these records. The rest of the outliers in our data seemed to be due to the rarity of some colors, breeds, or conditions (such as rabies).

We also explored our data and recognized a class imbalance issue between Euthanasia and the other three classes. To combat this, we used the imblearn library to perform random under-sampling and SMOTE in our models using the imblearn pipeline and compared these two methods to find our best model.

After the first attempts to run predictive models on the dataset, we decided to drop records for all classes besides Euthanasia and Adoption. Even after dropping records throughout our cleaning, classification models ran for extremely long amounts of time when used on the entire dataset — some were still running after more than an hour. Eliminating Transfer and Return to Owner also helped to alleviate the class imbalance, increasing the ratio of Euthanasia to all records. Despite only keeping Adoption and Euthanasia, we still had over 50,000 records in our dataset. 


Model Creation and Selection

To begin making and selecting a model, we looked at all supervised models we had experience using. Our dataset after cleaning and feature engineering was still of a very high dimensionality, so we expected models that fall apart at high dimensions, such as KNN, would be unfavorable. Our cleaned dataset was also rather large at over 50,000 rows after reduction. Thus, computationally expensive models would need to use a random sample of our data rather than the whole dataset. 

Once a set of models we were interested in was established, we created base models with little to no fine tuning in pursuit of a metric on model effectiveness. The models we ran on the data consisted of K-Nearest Neighbor, Naive Bayes, Decision Tree, Random Forest, Support Vector Machine (SVM), and Neural Net. After this step we were still unsure of which model to use, and decided to fine-tune each of these models.

We began to fine-tune, first by considering our imbalanced classes. For each of our models (besides random forest, which would not stop running with SMOTE) we tried one iteration where we passed SMOTE into our pipeline and one iteration where we passed random under-sampling into our pipeline. We tested different values for the ratio of our minority class to majority class using a parameter grid and GridSearchCV. We found that SMOTE typically gave us the best results by syntheticaly creating new records with Euthanasia labels, which are scarce in our dataset.

We also passed a standard scaler object into our pipeline for any model that needed scaling to work properly, like KNN. Additionally, for any model that needed dimensionality reduction, we passed principle component analysis into our pipeline. 

With our class imbalance, scaling, and dimensionality out of the way, we turned to tuning the various parameters for our chosen models, such as the max depth of a tree, using a parameter grid. We then passed a pipeline with all of these items—SMOTE or random under-sampling, scaling, PCA, and our chosen model—into a GridSearchCV, along with our various parameters in a parameter grid. This GridSearchCV was then passed into cross-val-predict. We tested various implementations and looked for a model that had a strong combination of both computational efficiency and a good score.

On scoring, because our dataset has a smaller number of Euthanasia rows than Adoption, even after SMOTE and under-sampling, we used F1 scores for our minority class to judge our models.


Challenges

One of the biggest challenges of this project was condensing the large data set into a manageable size without sacrificing important information. Eliminating records through cleaning and dropping certain outcomes alleviated this issue somewhat, but not as much as we had expected. Using multiple small samples of the data (five thousand records or less) solved the issue of the significantly long computing times, though we were not able to observe the performance of most of the predictive models on the entire data set.

Next, we faced a familiar issue: class imbalance. We did not have many records of Euthanasia compared to the other outcomes, likely due to the No Kill Implementation Plan currently in place. To solve the problem, we removed all Transfer and Return to Owner records, and we also used SMOTE or random undersampling. This significantly increased the accuracy scores of all classifiers. We then switched to using F1 scores, weighing precision and recall when determining which model was ideal for our data. Switching to using F1 scores worked well for solving this issue. We paid particular attention to the F1 score for Euthanasia as it was the minority class.

Finally, another challenge we realized towards the end of our notebook was that we had included attributes (Season and Age Upon Outcome) that did not make sense to include. These attributes should not have been included because they contain information that would be unknown until an outcome has occurred and therefore are not useful in predicting an animal’s outcome. Ideally, we would have retested all of our models after removing these columns, but because of the long run-time of our notebook, we decided to only redo our chosen model, decision trees.	


Results

To determine which model worked best for our data, we considered the minority (Euthanasia) class F1 scores. Although multiple models had the same highest F1 score for Euthanasia (0.94), we decided to use the decision tree model because it was the most computationally efficient and could be run on the entire data set without taking a significant amount of time.

After realizing the issues with Age Upon Outcome and Season, we decided to try our chosen model without these attributes to see if the F1 score was significantly impacted. Once we ran the decision tree model on the data set with Season and Age Upon Outcome removed, the F1 score for Euthanasia was still high at 0.93. 


Next Steps

Our model could be used to help predict the outcome for an animal that is either going to be adopted or euthanized. As described in our problem statement, this could help with various aspects of planning for animal shelters. It is also possible that the data could be manipulated and feature engineered such that not only would Adoption or Euthanasia be predicted, but all of the other outcomes as well. This would be much more practical and useful. 
Some areas that could be improved are the following:

More analysis could be done on which of our features to keep in our model in order to cut down on run-time. 
More analysis could be done on how best to feature engineer the Color and Breed column. It is possible the way we feature engineered was not the best option. 
It would have been more helpful for the shelter to record the age of the animal when they entered the shelter and the date at which they enter. 
It would be interesting to attempt to use our model to predict Adoption vs Transfer or Return to Owner, rather than just Adoption and Euthanasia. 

