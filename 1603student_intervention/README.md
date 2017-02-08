# Build a Student Intervention System

Supervised Learning Project

## Requirement

This project requires Python 2.7 and iPython/Jupyter Notebook.


## Data

The dataset used in this project is included as `student-data.csv`. It has the following atributes for each student:

- school - student's school (binary: "GP" or "MS")
- sex - student's sex (binary: "F" - female or "M" - male)
- age - student's age (numeric: from 15 to 22)
- address - student's home address type (binary: "U" - urban or "R" - rural)
- famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
- Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
- Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
- Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
- Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
- guardian - student's guardian (nominal: "mother", "father" or "other")
- traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
- studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
- failures - number of past class failures (numeric: n if 1<=n<3, else 4)
- schoolsup - extra educational support (binary: yes or no)
- famsup - family educational support (binary: yes or no)
- paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
- activities - extra-curricular activities (binary: yes or no)
- nursery - attended nursery school (binary: yes or no)
- higher - wants to take higher education (binary: yes or no)
- internet - Internet access at home (binary: yes or no)
- romantic - with a romantic relationship (binary: yes or no)
- famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
- freetime - free time after school (numeric: from 1 - very low to 5 - very high)
- goout - going out with friends (numeric: from 1 - very low to 5 - very high)
- Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
- Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
- health - current health status (numeric: from 1 - very bad to 5 - very good)
- absences - number of school absences (numeric: from 0 to 93)
- passed - did the student pass the final exam (binary: yes or no)



## References:http://rstudio-pubs-static.s3.amazonaws.com/4239_fcb292ade17648b097a9806fbe026e74.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
https://scholar.sun.ac.za/handle/10019.1/5360
http://research.microsoft.com/en-us/projects/decisionforests/
http://www.cs.cmu.edu/~qyj/papersA08/11-rfbook.pdf
http://www.ijcsi.org/papers/4-1-16-23.pdf
http://software.ucv.ro/~cmihaescu/ro/teaching/AIR/docs/Lab4-NaiveBayes.pdf
http://u.cs.biu.ac.il/~haimga/Teaching/AI/saritLectures/svm.pdf
http://www.distilnetworks.com/support-vector-machines-hadoop-theory-vs-practice/#.V1ciD5ErKhc
http://www.svms.org/finance/
http://www.eric-kim.net/eric-kim-net/posts/1/kernel_trick.html
https://panthimanshu17.wordpress.com/2013/07/28/svm-fundamentals-part-1/
http://www.support-vector.net/icml-tutorial.pdf

http://rstudio-pubs-static.s3.amazonaws.com/4239_fcb292ade17648b097a9806fbe026e74.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
https://scholar.sun.ac.za/handle/10019.1/5360
http://research.microsoft.com/en-us/projects/decisionforests/
http://www.cs.cmu.edu/~qyj/papersA08/11-rfbook.pdf
http://www.ijcsi.org/papers/4-1-16-23.pdf
http://software.ucv.ro/~cmihaescu/ro/teaching/AIR/docs/Lab4-NaiveBayes.pdf
http://u.cs.biu.ac.il/~haimga/Teaching/AI/saritLectures/svm.pdf
http://www.distilnetworks.com/support-vector-machines-hadoop-theory-vs-practice/#.V1ciD5ErKhc
http://www.svms.org/finance/
http://www.eric-kim.net/eric-kim-net/posts/1/kernel_trick.html
https://panthimanshu17.wordpress.com/2013/07/28/svm-fundamentals-part-1/
http://www.support-vector.net/icml-tutorial.pdf
