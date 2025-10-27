# API Server
this is a minimal flask API meant to serve as a runtime for our LLM since it would be difficult to run it in Isabelle/ML.

## Setup
All that is needed for setup is to install flask though pip. To do this change to this directory and run: 
`pip install -r requirements.txt`

## Running the server
To start the server locally run the following command:
`python app.py`


## TODOs
This API is currently only a wrapper around a dummy function which returns dummy templates. What we need to do is implement the following functionality
- Parse the function names we receive from the request into input for the LLM
- Run the LLM on the input and receive templates from the output
- Parse the templates into a JSON object and marshall it into a string that can be returned from the API



