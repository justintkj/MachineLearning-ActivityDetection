###imports
import sklearn

from SVM import SVMprocess
from KNN import KNNprocess
from NaiveBayes import NBprocess
from RandomForest import RFprocess

SVMtime,SVMacc = SVMprocess()
KNNtime, KNNacc = KNNprocess()
NBtime, NBacc = NBprocess()
RFtime, RFacc = RFprocess()
