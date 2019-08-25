from datasets.pascal_voc import pascal_voc

__sets = {}


for year in ['2007']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

def get_imdb(name):
    '''get an imdb(image dataset) by name'''
    if name not in __sets:
        raise KeyError('Unknow datasets: {}'.format(name))
    return __sets[name]()
    