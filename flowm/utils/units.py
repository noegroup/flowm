
def k_BT_in_kJ_per_mol(T=300):
    R = 0.00831446261815324 # Boltzmann constant in unit kJ/mol
    return R * T

def k_BT_in_kcal_per_mol(T=300):
    J_per_cal = 4.184
    return k_BT_in_kJ_per_mol(T) / J_per_cal

def inv_temp(T=300):
    """Returns inverse temperature in unit mol/kJ."""
    return 1. / k_BT_in_kJ_per_mol(T)

def inv_temp_mol_per_kcal(T=300):
    """Returns inverse temperature in unit mol/kcal."""
    return 1. / k_BT_in_kcal_per_mol(T)

