import csv

docs = []
with open('Shakespeare_data.csv') as csvfile:
    next(csvfile)
    with open('corpus',mode='w') as outfile:
        reader = csv.reader(csvfile)
        for row in reader:
            outfile.write(row[5]+'\n')
        
