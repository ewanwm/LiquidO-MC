# LiquidO-MC
Python code to run a basic simulation of a LiquidO \[[1]\] based detector

## Installation

Install via pip with the following command, from the project root directory
```
pip install .
```
It is recommended to use a virtual environment. You can create one using
```
python -m venv <env-name>
```
see more about venv [here](https://docs.python.org/3/library/venv.html)

## Usage
After installing, you can run the simulation using the main app `liquidOmc` from the command line.

Basic options are
`-n` - the number of photons to simulate
`-o` - the name of the output file to write the results to 

The following will simulate 10,000 photons and write their final states to "photons.csv"
```
liquidOmc -n 10000 -o "photons.csv"
```

run 
```
liquidOmc -h
```
to see all available configuration options


[1]: https://liquido.ijclab.in2p3.fr/
