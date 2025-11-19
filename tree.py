from __future__ import annotations
from typing import Callable, Generator, Optional, Self
from math import log
from dataclasses import dataclass

Value = int | float | str
Observation = list[Value]
Data = list[Observation]
Labels = list[Value]
ScoreFn = Callable[[Labels], float]

def unique_counts(values: list[Value]) -> dict[Value, int]:
    """Count how many times each value appears in `values`"""
    return {v : values.count(v) for v in values }

def is_numeric(value: Value) -> bool:
    """Checks if a value is numeric (i.e. a float or an int)"""
    return isinstance(value, int) or isinstance(value, float)

def get_query_fn(column: int, value: Value) -> Callable[[Observation], bool]:
    """
    Create a function that separates observations based on a query.
    The query can be:

    a) categorical: the created function returns true
       iff. the observation has the exact value in the column specified.
    b) continuous: the created function returns true
       iff. the observation has a value smaller than the reference one
       in the column specified.

    Note: consider any column with a numeric value as continuous.
    """
    if is_numeric(value):
        return lambda obs : obs[column] <= value
    else:
        return lambda obs : obs[column] == value


def unique_values(table: list[list[Value]], column_idx: int):
    """Returns a set of the values in the columns of a table."""
    values = set()
    for row in table:
        values.add(row[column_idx])
    return values

def cast_to(value_str: str) -> Value:
    """
    Given a value represented as a string, try to convert it
    to a more specific type (int, float) or fail back to string.
    """
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str

def log2(x): return log(x) / log(2)


@dataclass
class Node:
    column: Optional[int]
    value: Optional[Value]
    results: Optional[dict[Value, int]]
    true_branch: Optional[Node]
    false_branch: Optional[Node]

    def is_leaf(self):
        return self.true_branch is None
    
    @classmethod
    def new_node(cls, column, value, true_branch, false_branch) -> Node:
        """Create a new instance of this class representing a decision node."""
        return cls(column,value,None,true_branch,false_branch)

    @classmethod
    def new_leaf(cls, labels: Labels) -> Node:
        """Create a new instance of this class representing a leaf."""
        results = unique_counts(labels)
        return cls(None,None,results,None,None)

    def print_tree(self, indent=''):
        """Prints to stdout a representation of the tree."""
        # Is this a leaf node?
        if self.results!=None:
            print(self.results)
        else:
            # Print the criteria
            if is_numeric(self.value): #type:ignore
                print(f"{self.column}: >= {self.value}?")
            else:
                print(f"{self.column}: {self.value}?")
            # Print the branches
            print(f"{indent}T->", end="")
            self.true_branch.print_tree(indent+' ') #type:ignore
            print(f"{indent}F->", end="")
            self.false_branch.print_tree(indent+' ') #type:ignore
        
    def follow_tree(self, observation: Observation) -> Node:
        """
        Traverse the (sub)tree by answering the queries, until a leaf is reached.
        
        This method returns the leaf that this observation reaches.
        """

        current = self

        while not current.is_leaf():
            query = get_query_fn(current.column,current.value)
            if query(observation):
                current = current.true_branch
            else:
                current = current.false_branch
        return current


def _iterate_queries(observations : Data) -> Generator[tuple[int, Value],None,None]:
    assert len(observations) > 0, "No data"

    ncols = len(observations[0])
    for col in range(0, ncols):
        for value in unique_values(observations, col):
            yield col, value

def recursive_build_tree(scoref: ScoreFn, observations: Data, labels: Labels) -> Node:
    if not observations:
        return Node.new_leaf([])
    root_imp = scoref(labels)
    if root_imp == 0:
        return Node.new_leaf(labels)
    
    best_query, best_goodness, best_T, best_F = None, 0, None, None
    for col, value in _iterate_queries(observations):
        obs_true, labels_true, obs_false, labels_false = divideset(observations, labels, col, value)
        root = scoref(labels)
        tb = len(obs_true) / len(observations) * scoref(labels_true) 
        fb = len(obs_false) / len(observations) * scoref(labels_false)
        goodness = root - tb - fb
                                  
        if best_query is None or goodness > best_goodness:
                 best_query = col, value
                 best_goodness = goodness
                 best_T = obs_true, labels_true
                 best_F = obs_false, labels_false
    return Node.new_node(*best_query, recursive_build_tree(scoref, *best_T), recursive_build_tree(scoref, *best_F))
        
def divideset(
    observations: Data, labels: Labels, column: int, value: Value
) -> tuple[Data, Labels, Data, Labels]:
    """
    Divides a set on a specific column.
    Can handle numeric or categorical values
    """                              
                                  
    query_fn = get_query_fn(column, value)
    
    observations_true, labels_true, observations_false, labels_false = [], [], [], []

    for obv, label in zip(observations, labels):
        if query_fn(obv):
            observations_true.append(obv)
            labels_true.append(label)
        else:
            observations_false.append(obv)
            labels_false.append(label)

    return observations_true, labels_true, observations_false, labels_false

def gini(labels: Labels) -> float:
    total = len(labels)
    results = unique_counts(labels)
    probs={ label : count/total for label,count in results.items() }
    return 1 - sum(p**2 for p in probs.values())

def entropy(labels):
    total = len(labels)
    results = unique_counts(labels)
    probs = {
        label: count / total for label, count in results.items()
    }

    return -sum( p* log2(p) for p in probs.values())


class DecisionTreeModel:
    def __init__(self, scoref: ScoreFn = gini, beta: float = 0, prune_threshold: float = 0):
        self.scoref = scoref
        self.beta = beta
        self.prune_threshold = prune_threshold
    
    def fit(self, observations: Data, labels: Labels) -> Self:
        self.tree_ = recursive_build_tree(self.scoref, observations, labels)
        return self
    
    def predict(self, observations: Data) -> Labels:
        labels=[]
        for observation in observations:
            leaf = self.tree_.follow_tree(observation)
            results = leaf.results
            label = max(results.keys(), key=results.get)
            labels.append(label)
        return labels

    def score(self, data: Data, labels: Labels) -> float:
        predicted = self.predict(data)
        correct = sum(
            1 if pred == expected else 0
            for pred, expected in zip(predicted, labels)
        )
        return correct / len(data)

def read_csv(file_name: str) -> list[list[Value]]:
    table = []
    with open(file_name) as f:
        for line in f:
            row = line.split(',')
            table.append(row)
    return table


def split_observations_and_labels(table: list[list[Value]]) -> tuple[Data, Labels]:
    data, labels = [], []
    for row in table:
        data.append(row[:-1])
        labels.append(row[-1])
    return data, labels
