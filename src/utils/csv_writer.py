import pandas as pd

""""
This class is used to write data to a csv file.
"""
class CSVWriter:
    def __init__(self, filename, column_names, data_values, header=None):
        self.filename = filename + '.csv'
        self.header = header
        self.writer = None
        self.column_names = column_names
        self.data_values = data_values

    def create_csv_file(self):
        self.data_values.insert(0, self.column_names)
        self.writer = pd.DataFrame(data=self.data_values)
        self.writer.to_csv(self.filename, header=self.header, index=False)


if __name__ == "__main__":
    CSV = CSVWriter("../tmp",
                    ["Fruit", "Color", "Price"],
                    [["Apple", "Red", 5.0],
                     ["Orange", "Orange", 3.0],
                     ["Apple", "Green", 4.0],
                     ["Mango", "Orange - Red", 7.0],
                     ["Kiwi", "Green", 2.0],
                     ])
    CSV.create_csv_file()
