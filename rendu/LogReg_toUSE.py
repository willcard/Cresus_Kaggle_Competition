from nettoyage import *
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV,train_test_split

#           #
#   Train   #
#           #

FILTRER = True

train = pd.read_csv('../data/train.csv')
train = mega_traitement(train,dummies=False)

or_dummies = list(train.columns[-6:])
to_drop = ['Date','id'] #+ or_dummies

X = train.drop(columns=to_drop +['ORIENTATION'])
y = train[['ORIENTATION']]


# # Sélection de features probablement utiles
# to_use =['REVENUS','CHARGES','CREDITS','moy_eco_jour','PROF','situation',
#         'LOGEMENT','pers_a_charge','cat_RAV_UC','NATURE_DIFF',
#         'PLATEFORME','nb_credits','nb_immo','nb_rac','nb_autres','nb_decouvert',
#         'RAV_ouverture', 'crd_renouv', 'crd_amort',]
to_use =['CRD','REVENUS','CHARGES','CREDITS','moy_eco_jour','RAV_ouverture', 'crd_renouv', 'crd_amort']

if FILTRER:
    __X = X[to_use]
else:
# __X = myAcp(X)
__X = X

X_train, X_test, y_train, y_test = train_test_split(__X, y, test_size=0.2, random_state=42)

LogReg = LogisticRegression(solver='newton-cg', multi_class='multinomial', random_state=42)
LogReg.fit(X_train,y_train)
print(f'>> Score local:{LogReg.score(X_test,y_test)}')

#          #
#   Test   #
#          #

test = pd.read_csv('../data/test.csv')
test = mega_traitement(test,dummies=False)
#test_ = test.drop(columns=['Date','id'])
test_ = test[to_use]

submission = pd.DataFrame()
submission['id'] = test.id
submission['ORIENTATION'] = LogReg.predict(test_)

#                #
#   Soumission   #
#                #

dict_orienta = {5:'Surendettement',3:'Mediation',
                    0:'Accompagnement',1:'Aucune',
                   4:'Microcredit',2:'Autres Procédures Collectives'}
submission.ORIENTATION = submission.ORIENTATION.replace(dict_orienta)

print(submission.head(10))

submission.to_csv('LogRegSubmission_toUSE.csv',index=False)
