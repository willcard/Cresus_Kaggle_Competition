from nettoyage import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split

#           #
#   Train   #
#           #

train = pd.read_csv('../data/train.csv')
train = mega_traitement(train,dummies=False)

or_dummies = list(train.columns[-6:])
to_drop = ['Date','id'] #+ or_dummies

X = train.drop(columns=to_drop +['ORIENTATION'])
y = train[['ORIENTATION']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf_best = RandomForestClassifier(max_depth=75, bootstrap=True, max_features=8, min_samples_leaf=2, min_samples_split=12, n_estimators=300)
clf_best.fit(X_train, y_train)

#          #
#   Test   #
#          #

test = pd.read_csv('../data/test.csv')
test = mega_traitement(test,dummies=False)
test_ = test.drop(columns=['Date','id'])

submission = pd.DataFrame()
submission['id'] = test.id
submission['ORIENTATION'] = clf_best.predict(test_)

#                #
#   Soumission   #
#                #

dict_orienta = {5:'Surendettement',3:'Mediation',
                    0:'Accompagnement',1:'Aucune',
                   4:'Microcredit',2:'Autres Procédures Collectives'}
submission.ORIENTATION = submission.ORIENTATION.replace(dict_orienta)

print(submission.head(10))

submission.to_csv('NewSubmission.csv',index=False)
