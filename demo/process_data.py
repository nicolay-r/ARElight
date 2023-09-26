#!/usr/bin/python3
# Above shebang makes sure we use Python 3

import cgi
import cgitb
cgitb.enable()  # Enable debugging - useful for development but should be disabled in production

def main():
    form = cgi.FieldStorage()

    # Fetching the input data (assuming textarea's name attribute is 'data')
    input_data = form.getvalue('data', 'No data provided')

    # Processing the data (in this case just echoing it with a message)
    output = "Processed: " + input_data

    print("Content-type: text/plain\n")
    print(output)

if __name__ == "__main__":
    main()
