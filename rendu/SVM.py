from nettoyage import *

FILTRER = True

train = pd.read_csv('../data/train.csv')
train = mega_traitement(train,dummies=False)

or_dummies = list(train.columns[-6:])
to_drop = ['Date','id'] #+ or_dummies

X = train.drop(columns=to_drop +['ORIENTATION'])
_X = myAcp(X)
y = train[['ORIENTATION']]
