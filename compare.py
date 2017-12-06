with open("C:/Users/owner/PycharmProjects/finalProject/tmp/census_data/numbers.txt", "r") as ins:
    compare = []
    numbers = ''
    for line in ins:

        compare = line.split(',')
        print (compare[0], compare[1])
        this = int(compare[0])
        last = int(compare[1])
        if (this > last) :
            numbers = numbers+'1,'

        else:
            numbers = numbers+'0,'
    print (numbers)
    ins.close()
