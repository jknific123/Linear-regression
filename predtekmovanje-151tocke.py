import csv
import gzip
import numpy
from sklearn import linear_model
import linear
import lpputils

if __name__ == "__main__":

    prazniki = [(1,1), (2,1), (8,2), (8,4), (9,4), (27,4), (1,5), (2,5), (25,6), (15,8), (31,10), (1,11), (25,12), (26,12)]


    # train podatki
    f = gzip.open("train_pred.csv.gz", "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter="\t")
    next(reader) # ne rabimo prve vrstice
    train_data = [ d for d in reader ]

    # OBDELAVA PODATKOV IN IZDELAVA X
    matrika = []
    matrika2 = []
    for voznja in train_data:


        vrstica_matrike = numpy.zeros(10)
        odhod = lpputils.parsedate(voznja[-3])

        # za y
        matrika2.append(lpputils.tsdiff(voznja[-1], voznja[-3]))

        # dan v tednu
        vrstica_matrike[0] = odhod.isoweekday()

        # vikend da/ne
        vikend = odhod.isoweekday()
        if (vikend in (6,7)):
            vrstica_matrike[1] = 1
        else:
            vrstica_matrike[1] = 0

        # ura odhoda
        ura = odhod.hour
        vrstica_matrike[2] = ura

        # praznik da/ne
        dan = odhod.day
        mesec = odhod.month
        for praznik in prazniki:

            if (praznik[0] == dan and praznik[1] == mesec):
                vrstica_matrike[3] = 1
            else:
                vrstica_matrike[3] = 0

        # mesec
        vrstica_matrike[4] = mesec

        if (ura >= 6 and ura < 10):
            vrstica_matrike[5] = 1
        elif (ura > 10 and ura < 14):
            vrstica_matrike[6] = 1
        elif (ura >= 14 and ura < 18):
            vrstica_matrike[7] = 1
        elif (ura >= 18 and ura < 21):
            vrstica_matrike[8] = 1
        elif (ura >= 21 and ura < 6):
            vrstica_matrike[9] = 1

        matrika.append(vrstica_matrike)

    X = numpy.vstack(matrika)
    Y = numpy.array(matrika2)
    # print(X)
    # print(Y)

    # TESTNI PODATKI DEL

    # testni podatki
    f = gzip.open("test_pred.csv.gz", "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter="\t")
    next(reader) # ne rabimo prve vrstice
    test_data = [ d for d in reader ]


    testna_matrika = []
    for voznja in test_data:

        vrstica_matrike2 = numpy.zeros(10)
        odhod = lpputils.parsedate(voznja[-3])

        # dan v tednu
        vrstica_matrike2[0] = odhod.isoweekday()

        # vikend da/ne
        vikend = odhod.isoweekday()
        if (vikend in (6,7)):
            vrstica_matrike2[1] = 1
        else:
            vrstica_matrike2[1] = 0

        # ura odhoda
        ura = odhod.hour
        vrstica_matrike2[2] = ura

        # praznik da/ne
        dan = odhod.day
        mesec = odhod.month
        for praznik in prazniki:

            if (praznik[0] == dan and praznik[1] == mesec):
                vrstica_matrike2[3] = 1
            else:
                vrstica_matrike2[3] = 0

        # mesec
        vrstica_matrike2[4] = mesec

        # deli dneva
        if (ura >= 6 and ura < 10):
            vrstica_matrike2[5] = 1
        elif (ura > 10 and ura < 14):
            vrstica_matrike2[6] = 1
        elif (ura >= 14 and ura < 18):
            vrstica_matrike2[7] = 1
        elif (ura >= 18 and ura < 21):
            vrstica_matrike2[8] = 1
        elif (ura >= 21 and ura < 6):
            vrstica_matrike2[9] = 1

        testna_matrika.append(vrstica_matrike2)

    TestMatrix = numpy.vstack(testna_matrika)
    # print(TestMatrix)

    # linearna regresija

    reg = linear_model.LinearRegression()
    reg.fit(X, Y)

    rez = reg.predict(TestMatrix)


    f = gzip.open("test_pred.csv.gz", "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter="\t")
    next(reader)  # skip legend

    fo = open("linRegRezultatiCeloLeto+deliDneva.txt", "wt")
    i = 0
    for l in reader:
        fo.write(lpputils.tsadd(l[-3], rez[i]) + "\n")
        i += 1
    fo.close()
    
