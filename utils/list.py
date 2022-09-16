
def indices_of(list:list,value)->list[int]:
    indices =[]
    for i,l in enumerate(list):
        if value == l:
            indices.append(i)
    return indices

def get_all(list:list,indices:list[int])->list:
    return [list[i] for i in indices]
