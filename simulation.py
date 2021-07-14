"""
Simulate data for training or testing using msprime.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang
Date: 2/4/21
"""

# python imports
import math
import msprime
import numpy as np
import random
import sys
import time
from collections import OrderedDict

from numpy.random import default_rng

# from stdpopsim
import sps.engines
import sps.species
import sps.HomSap

# our imports
import real_data_random
import util

################################################################################
# SIMULATION
################################################################################

class Generator:

    def __init__(self, simulator, param_names, sample_sizes, num_snps, L, seed,\
        mirror_real=False, reco_folder="", filter=False):
        self.simulator = simulator
        self.param_names = param_names
        self.sample_sizes = sample_sizes
        self.num_samples = sum(sample_sizes)
        self.num_snps = num_snps
        self.L = L
        self.rng = default_rng(seed)
        self.curr_params = None

        # for real data, use HapMap. this also turns on singleton filtering
        self.prior = []
        self.weights = []
        self.filter = filter # for singletons
        if mirror_real and reco_folder != None:
            files = [reco_folder + "genetic_map_chr" + str(i) + ".txt" \
                for i in [2, 3]]
            self.prior, self.weights = util.parse_hapmap_empirical_prior(files)

    def simulate_batch(self, batch_size, params=[], real=False, neg1=True):

        # initialize 4D matrix (two channels for distances)
        if self.num_snps == None:
            regions = []
        else:
            regions = np.zeros((batch_size, self.num_samples, self.num_snps, \
                2), dtype=np.float32) # two channels

        # set up parameters
        sim_params = util.ParamSet()
        if real:
            pass # keep orig for "fake" real
        elif params == []:
            sim_params.update(self.param_names, self.curr_params)
        else:
            sim_params.update(self.param_names, params)

        # simulate each region
        for i in range(batch_size):
            seed = self.rng.integers(1,high=2**32) # like GAN "noise"

            ts = self.simulator(sim_params, self.sample_sizes, self.L, seed, \
                prior=self.prior, weights=self.weights)
            region = prep_region(ts, self.num_snps, self.L, self.filter, neg1)

            if self.num_snps == None:
                regions.append(region)
            else:
                regions[i] = region

        return regions

    def real_batch(self, batch_size, is_train): # ignore is_train
        return self.simulate_batch(batch_size, real=True)

    def update_params(self, new_params):
        self.curr_params = new_params

def draw_background_rate_from_prior(prior_rates, prob):
    return np.random.choice(prior_rates, p=prob)

def prep_region(ts, num_snps, L, filter, neg1):
    """Gets simulated data ready"""
    gt_matrix = ts.genotype_matrix().astype(float)
    snps_total = gt_matrix.shape[0]

    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    dist_vec = [0] + [(positions[j+1] - positions[j])/L for j in \
        range(snps_total-1)]

    # when mirroring real data
    if filter:
        return util.process_gt_dist(gt_matrix, dist_vec, num_snps, filter=True,\
            rate=0.3, neg1=neg1)
    else:
        return util.process_gt_dist(gt_matrix, dist_vec, num_snps, neg1=neg1)

def simulate_im(params, sample_sizes, L, seed, prior=[], weights=[]):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

    # sample reco or use value
    if prior != []:
        reco = draw_background_rate_from_prior(prior, weights)
    else:
        reco = params.reco.value

    # condense params
    N1 = params.N1.value
    N2 = params.N2.value
    T_split = params.T_split.value
    N_anc = params.N_anc.value
    mig = params.mig.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0], \
            initial_size = N1),
        msprime.PopulationConfiguration(sample_size=sample_sizes[1], \
            initial_size = N2)]

    # no migration initially
    mig_time = T_split/2

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 1, \
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 0, \
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
		# move all in deme 1 to deme 0
		msprime.MassMigration(
			time = T_split, source = 1, destination = 0, proportion = 1.0),
        # change to ancestral size
        msprime.PopulationParametersChange(time=T_split, initial_size=N_anc, \
            population_id=0)
	]

    # simulate tree sequence
    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length = L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def simulate_ooa2(params, sample_sizes, L, seed, prior=[], weights=[]):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

    # sample reco or use value
    if prior != []:
        reco = draw_background_rate_from_prior(prior, weights)
    else:
        reco = params.reco.value

    # condense params
    T1 = params.T1.value
    T2 = params.T2.value
    mig = params.mig.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0], \
            initial_size = params.N3.value), # YRI is first
        msprime.PopulationConfiguration(sample_size=sample_sizes[1], \
            initial_size = params.N2.value)] # CEU/CHB is second

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = T2, source = 1, \
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = T2, source = 0, \
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
        # change size of EUR
        msprime.PopulationParametersChange(time=T2, \
            initial_size=params.N1.value, population_id=1),
		# move all in deme 1 to deme 0
		msprime.MassMigration(time = T1, source = 1, destination = 0, \
            proportion = 1.0),
        # change to ancestral size
        msprime.PopulationParametersChange(time=T1, \
            initial_size=params.N_anc.value, population_id=0)
	]

    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length = L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def simulate_postOOA(params, sample_sizes, L, seed, prior=[], weights=[]):
    """Note this is a 2 population model for CEU/CHB split"""
    assert len(sample_sizes) == 2

    # sample reco or use value
    if prior != []:
        reco = draw_background_rate_from_prior(prior, weights)
    else:
        reco = params.reco.value

    # condense params
    T1 = params.T1.value
    T2 = params.T2.value
    mig = params.mig.value
    #m_EU_AS = params.m_EU_AS.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0], \
            initial_size = params.N3.value), # CEU is first
        msprime.PopulationConfiguration(sample_size=sample_sizes[1], \
            initial_size = params.N2.value)] # CHB is second

    # symmetric migration
    #migration_matrix=[[0, m_EU_AS],
    #                  [m_EU_AS, 0]]

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = T2/2, source = 1, \
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = T2/2, source = 0, \
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
		# move all in deme 1 to deme 0
		msprime.MassMigration(time = T2, source = 1, destination = 0, \
            proportion = 1.0),
        # set mig rate to zero (need if using migration_matrix)
        #msprime.MigrationRateChange(time=T2, rate=0),
        # ancestral bottleneck
        msprime.PopulationParametersChange(time=T2, \
            initial_size=params.N1.value, population_id=0),
        # ancestral size
        msprime.PopulationParametersChange(time=T1, \
            initial_size=params.N_anc.value, population_id=0)
	]

    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
        #migration_matrix = migration_matrix,
		mutation_rate = params.mut.value,
		length = L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def simulate_KF(params, sample_sizes, L, seed, prior=[], weights=[]):
    """Note this is a 2 population model for Kir/Fol split"""
    assert len(sample_sizes) == 2
    
    # sample reco or use value
    if prior != []:
        reco = draw_background_rate_from_prior(prior, weights)
    else:
        reco = params.reco.value

    # condense params
    N1 = params.N1.value
    N2 = params.N2.value

    # pop1
    T_N1_1 = params.T_N1_1.value
    N1_1 = params.N1_1.value
    T_N1_2 = params.T_N1_2.value
    N1_2 = params.N1_2.value
    T_N1_3 = params.T_N1_3.value
    N1_3 = params.N1_3.value
    # pop2    
    T_N2_1 = params.T_N2_1.value
    N2_1 = params.N2_1.value
    T_N2_2 = params.T_N2_2.value
    N2_2 = params.N2_2.value
    T_N2_3 = params.T_N2_3.value
    N2_3 = params.N2_3.value
    # migration
    mig = params.mig.value
    T_mig = params.T_mig.value
    mig1 = params.mig1.value
    T_mig1 = params.T_mig1.value
    mig2 = params.mig2.value
    T_mig2 = params.T_mig2.value
    T_mig0 = params.T_mig0.value
    # split       
    T_split = params.T_split.value
    N_anc = params.N_anc.value
    # ancestral pop size
    T_N_anc1 = params.T_N_anc1.value
    N_anc1 = params.N_anc1.value
    T_N_anc2 = params.T_N_anc2.value
    N_anc2 = params.N_anc2.value
    
    time_dt = {T_N1_1: ["Ne0", N1_1], T_N1_2: ["Ne0", N1_2], T_N1_3: ["Ne0", N1_3],
               T_N2_1: ["Ne1", N2_1], T_N2_2: ["Ne1", N2_2], T_N2_3: ["Ne1", N2_3],
               T_mig: ["mig01", mig], T_mig1: ["mig01", mig1], T_mig2: ["mig01", mig2], T_mig0: ["mig01", 0],
               T_split: ["div", N_anc], T_N_anc1: ["Ne01", N_anc1], T_N_anc2: ["Ne01", N_anc2]
               }
    
    od = OrderedDict(sorted(time_dt.items()))
    
    asymmetric_mig = False
    # start pop config/demo
    sample_sizes_diploid = {f"pop_{i}":s/2 for i, s in enumerate(sample_sizes)}
    demo_events = msprime.Demography()
    for i, init in enumerate([N1, N2]):
        demo_events.add_population(name=f"pop_{i}", initial_size=init)

    source_list = [] 
    for times in od.keys():
        event, value = od[times]
        if "Ne" in event:
            pop_id = event.lstrip("Ne")
            if f"pop_{pop_id}" not in source_list:
                demo_events.add_population_parameters_change(time=times, initial_size=value, population=f"pop_{pop_id}", growth_rate=0)
        elif "mig" in event:
            pop_id = list(event.lstrip("mig"))
            if all(f"pop_{i}" not in source_list for i in pop_id):
                if asymmetric_mig:
                    # asym
                    if value > 0:
                        # migration from pop 1 into pop 0 (back in time)
                        demo_events.add_migration_rate_change(time=times, source="pop_1" , dest="pop_0", rate=value)
                    elif value < 0:
                        # migration from pop 0 into pop 1 (back in time)
                        demo_events.add_migration_rate_change(time=times, source="pop_0" , dest="pop_1", rate=abs(value))
                    elif value == 0:
                        demo_events.add_symmetric_migration_rate_change(time=times, populations=pop_id, rate=value)
                else:
                    # sym
                    demo_events.add_symmetric_migration_rate_change(time=times, populations=pop_id, rate=value)

        elif "div" in event:
            demo_events.add_population(name="pop_01", initial_size=value)
            demo_events.add_population_split(time=times, derived=["pop_0", "pop_1"], ancestral="pop_01")
            source_list.append("pop_0")
            source_list.append("pop_1")

    tree = msprime.sim_ancestry(sample_sizes_diploid,
                                recombination_rate=reco,
                                demography=demo_events,
                                sequence_length=L,
                                random_seed = seed)
    ts = msprime.sim_mutations(tree,
                               rate=params.mut.value,
                               model="InfiniteAlleles",
                               random_seed = seed)

    return ts

def simulate_exp(params, sample_sizes, L, seed, prior=[], weights=[]):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    # sample reco or use value
    if prior != []:
        reco = draw_background_rate_from_prior(prior, weights)
    else:
        reco = params.reco.value

    T2 = params.T2.value
    N2 = params.N2.value

    N0 = N2 / math.exp(-params.growth.value * T2)

    demographic_events = [
        msprime.PopulationParametersChange(time=0, initial_size=N0, \
            growth_rate=params.growth.value),
        msprime.PopulationParametersChange(time=T2, initial_size=N2, \
            growth_rate=0),
		msprime.PopulationParametersChange(time=params.T1.value, \
            initial_size=params.N1.value)
	]

    ts = msprime.simulate(sample_size = sum(sample_sizes),
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length = L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def simulate_const(params, sample_sizes, L, seed, prior=[], weights=[]):
    assert len(sample_sizes) == 1

    # sample reco or use value
    if prior != []:
        reco = draw_background_rate_from_prior(prior, weights)
    else:
        reco = params.reco.value

    # simulate data
    ts = msprime.simulate(sample_size=sum(sample_sizes), Ne=params.Ne.value, \
        length=L, mutation_rate=params.mut.value, recombination_rate=reco, \
        random_seed = seed)

    return ts

def simulate_ooa3(params, sample_sizes, L, seed, prior=[], weights=[]):
    """From OOA3 as implemented in stdpopsim"""
    assert len(sample_sizes) == 3

    sp = sps.species.get_species("HomSap")

    mult = L/141213431 # chr9
    contig = sp.get_contig("chr9",length_multiplier=mult) # TODO vary the chrom

    # 14 params
    N_A = params.N_A.value
    N_B = params.N_B.value
    N_AF = params.N_AF.value
    N_EU0 = params.N_EU0.value
    N_AS0 = params.N_AS0.value
    r_EU = params.r_EU.value
    r_AS = params.r_AS.value
    T_AF = params.T_AF.value
    T_B = params.T_B.value
    T_EU_AS = params.T_EU_AS.value
    m_AF_B = params.m_AF_B .value
    m_AF_EU = params.m_AF_EU.value
    m_AF_AS = params.m_AF_AS.value
    m_EU_AS = params.m_EU_AS.value

    model = sps.HomSap.ooa_3(N_A, N_B, N_AF, N_EU0, N_AS0, r_EU, r_AS, T_AF, \
        T_B, T_EU_AS, m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS)
    samples = model.get_samples(sample_sizes[0], sample_sizes[1], \
        sample_sizes[2]) #['YRI', 'CEU', 'CHB']
    engine = sps.engines.get_engine('msprime')
    ts = engine.simulate(model, contig, samples)

    return ts

# testing
if __name__ == "__main__":

    batch_size = 50
    S = 36
    R = 50000
    SEED = 1833
    params = util.ParamSet()

    # quick test
    print("sim exp")
    generator = Generator(simulate_exp, ["N1", "T1"], [20], S, R, SEED)
    generator.update_params([params.N1.value, params.T1.value])
    mini_batch = generator.simulate_batch(50)
    print("x", mini_batch.shape)
