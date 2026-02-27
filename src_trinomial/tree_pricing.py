import time 
from datetime import date

from src_trinomial.tree import Tree
from src_trinomial.trinomial_model import TrinomialModel

from src.option_trade import OptionTrade
from src.market import Market

def tree_pricing(market : Market, option : OptionTrade, date_valuation : date, nb_step = 500):
    prunning_threshold = 1e-8 #1e-15 par exemple 
    tree = Tree(nb_step, market, option, date_valuation, prunning_threshold)
    mod = TrinomialModel(date_valuation, tree)
    print("Tree created with root underlying:", tree.root.underlying_i)
    
    start = time.time()
    tree.build_tree()
    end = time.time()

    print("temps d'exécution création arbre:", end - start, "secondes")
    
    start = time.time()
    #print("L'option via pricing recursif vaut = ", mod.price(option, "recursive"))
    print("L'option via backward induction vaut = ", mod.price(option, "backward"))
    end = time.time()
    print("temps d'exécution priceur récursif:", end - start, "secondes")

    #tree.plot_tree()
    # Greeks

    print("Delta =", mod.delta(option))
    print("Gamma =", mod.gamma(option))
    print("Delta hedge", 1000*mod.delta(option))
    if nb_step <= 950: #car utilise price recursive qui est limité à 950 
        
        print("Vega  =", mod.vega(option))
        print("Vomma =", mod.vomma(option))
        print("Vanna =", mod.vanna(option))