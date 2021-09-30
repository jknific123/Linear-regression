import gzip
import numpy
import csv
import lpputils
import linear
from sklearn import linear_model

from collections import defaultdict

def linekey(d):
    return tuple(d[2:4])


def casPoti(podatki):

    matrika2 = []
    for voznja in podatki:
        # cas voznje
        matrika2.append(lpputils.tsdiff(voznja[8], voznja[6]))

    rezY = numpy.array(matrika2)
    return rezY


def atributi(podatki):

    matrika = []

    #print(podatki)

    for voznja in podatki:

        vrstica_matrike = numpy.zeros(40)
        #print(voznja)
        odhod = lpputils.parsedate(voznja[6])

        # ure
        ura = odhod.hour
        vrstica_matrike[ura] = 1

        # dnevi v tednu
        dan = odhod.isoweekday()
        vrstica_matrike[23 + dan] = 1

        # deli dneva
        if (ura >= 6 and ura < 10):
            vrstica_matrike[31] = 1
        elif (ura > 10 and ura < 14):
            vrstica_matrike[32] = 1
        elif (ura >= 14 and ura < 18):
            vrstica_matrike[34] = 1
        elif (ura >= 18 and ura < 21):
            vrstica_matrike[35] = 1
        elif (ura >= 21 and ura < 6):
            vrstica_matrike[36] = 1

        # vikend da/ne
        if (dan in (6, 7)):
            vrstica_matrike[37] = 1

        # praznik da/ne
        mesec = odhod.month
        for praznik in prazniki:

            if (praznik[0] == dan and praznik[1] == mesec):
                vrstica_matrike[38] = 1

        # mesec
        vrstica_matrike[39] = mesec

        matrika.append(vrstica_matrike)

    rezX = numpy.vstack(matrika)
    return rezX


def narediSlovar(data, slovarLinij, napovedniki):


    # locimo razlicne avtobusne linije vsaka linija je predstavljena v slovatrju key = linija, value = list of lists -> vsi zapisi te linije
    for d in data:
        #print(linekey(d))
        casVoznje = lpputils.tsdiff(d[8], d[6])
        if (d[6] < d[8]): # odhod more bit manjsi od prihoda
            slovarLinij[linekey(d)].append(d)

    # za vsako linijo zgradimo napovedni model in ga shranimo v slovar modelov
    for linija in slovarLinij:
        # print(linija)
        lr = linear.LinearLearner(lambda_=1.)
        X = atributi(slovarLinij[linija])
        Y = casPoti(slovarLinij[linija])
        napoved = lr(X,Y)
        #reg = linear_model.LinearRegression().fit(X, Y)
        # reg.fit(X, Y)
        napovedniki[linija] = napoved


prazniki = [(1, 1), (2, 1), (8, 2), (8, 4), (9, 4), (27, 4), (1, 5), (2, 5), (25, 6), (15, 8), (31, 10), (1, 11),
            (25, 12), (26, 12)]

slovarLinij = defaultdict(list)
napovedniki = {}

if __name__ == "__main__":

    f = gzip.open("train.csv.gz", "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    data = [ d for d in reader ]

    narediSlovar(data, slovarLinij, napovedniki)
    #print(slovarLinij[('1', 'MESTNI LOG - VIŽMARJE', '  VIŽMARJE')])

    f = gzip.open("test.csv.gz", "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter="\t")
    next(reader) #skip legend
    test_data = [d for d in reader]


    #TestMatrix = numpy.vstack(testna_matrika)
    testna_matrika = atributi(test_data)
    rez = []
    for i in range(len(testna_matrika)):
        #rezultatSklearn[i] = napovedniki[linekey(test_data[i])].predict(testna_matrika[i])
        value = napovedniki[linekey(test_data[i])](testna_matrika[i])
        rez.append(value)

    fo = open("linRegRezultatiTekmovanje2.txt", "wt")
    i = 0
    for l in test_data:
        #print(l[-3], rez[i])
        fo.write(lpputils.tsadd(l[6], rez[i]) + "\n")
        i += 1
    fo.close()