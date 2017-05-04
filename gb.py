import itertools as iter

def pset(lst):
    comb = (iter.combinations(lst, l) for l in range(len(lst) + 1))
    return list(iter.chain.from_iterable(comb))

features = ['dotsInPP','foursInPP','numOfDots','numOfFours']
print(features[0])
print(pset(features)[1])