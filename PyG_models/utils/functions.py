import torch

def count_funct_groups(data):
    '''
    '''

    # Empty list to store numbers
    numbers = []
    oh = 0
    cooh = 0
    keto = 0
    cho = 0
    epoxy = 0
    for x in data:
        numbers = []
        # Extract numbers from mol name string
        for char in x.mol:
            if char.isdigit():
                numbers.append(int(char))

        oh += numbers[0]
        cooh += numbers[1]
        epoxy += numbers[2]
        cho += numbers[3]
        keto += numbers[4]
        oh += numbers[5]
        epoxy += numbers[6]
        epoxy += numbers[7]

    tot_groups = oh + cooh + epoxy + cho + keto

    oh_perc = oh / tot_groups
    cooh_perc = cooh / tot_groups
    epoxy_perc = epoxy / tot_groups
    cho_perc = cho / tot_groups
    keto_perc = keto / tot_groups
    
    return oh_perc*100, cooh_perc*100, epoxy_perc*100, cho_perc*100, keto_perc*100   

def RSE_loss(prediction, target):
    '''
    '''

    dE = (300 - 280) / 200
    nom = torch.sum(dE*torch.pow((target-prediction), 2))
    denom = torch.sum(dE*target)
    return torch.sqrt(nom) / denom