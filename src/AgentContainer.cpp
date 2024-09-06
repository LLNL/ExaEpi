/*! @file AgentContainer.cpp
    \brief Function implementations for #AgentContainer class
*/

#include "AgentContainer.H"

using namespace amrex;


/*! Add runtime SoA attributes */
void AgentContainer::add_attributes()
{
    const bool communicate_this_comp = true;
    {
        int count(0);
        for (int i = 0; i < m_num_diseases*RealIdxDisease::nattribs; i++) {
            AddRealComp(communicate_this_comp);
            count++;
        }
        Print() << "Added " << count << " real-type run-time SoA attibute(s).\n";
    }
    {
        int count(0);
        for (int i = 0; i < m_num_diseases*IntIdxDisease::nattribs; i++) {
            AddIntComp(communicate_this_comp);
            count++;
        }
        Print() << "Added " << count << " integer-type run-time SoA attibute(s).\n";
    }
    return;
}

/*! Constructor:
    *  + Initializes particle container for agents
    *  + Read in contact probabilities from command line input file
    *  + Read in disease parameters from command line input file
*/
AgentContainer::AgentContainer (const amrex::Geometry            & a_geom,  /*!< Physical domain */
                                const amrex::DistributionMapping & a_dmap,  /*!< Distribution mapping */
                                const amrex::BoxArray            & a_ba,    /*!< Box array */
                                const int                        & a_num_diseases, /*!< Number of diseases */
                                const std::vector<std::string>   & a_disease_names /*!< names of the diseases */)
    : amrex::ParticleContainer< 0,
                                0,
                                RealIdx::nattribs,
                                IntIdx::nattribs> (a_geom, a_dmap, a_ba),
        m_student_counts(a_ba, a_dmap, SchoolType::total_school_type, 0),
        reset_school_infection(a_ba, a_dmap, SchoolType::nattribs, 0)
{
    BL_PROFILE("AgentContainer::AgentContainer");

    m_num_diseases = a_num_diseases;
    AMREX_ASSERT(m_num_diseases < ExaEpi::max_num_diseases);
    m_disease_names = a_disease_names;

    m_student_counts.setVal(0);
    reset_school_infection.setVal(0);
    add_attributes();

    {
        amrex::ParmParse pp("agent");
        pp.query("symptomatic_withdraw", m_symptomatic_withdraw);
        pp.query("shelter_compliance", m_shelter_compliance);
        pp.query("symptomatic_withdraw_compliance", m_symptomatic_withdraw_compliance);
        pp.queryarr("student_teacher_ratios", m_student_teacher_ratios);
        pp.query("sc_infection_threshold", m_sc_infection_threshold);
        pp.query("sc_period", m_sc_period);

    }

    {
        using namespace ExaEpi;

        /* Create the interaction model objects and push to container */
        m_interactions.clear();
        m_interactions[InteractionNames::generic] = new InteractionModGeneric<PCType,PTileType,PTDType,PType>;
        m_interactions[InteractionNames::home] = new InteractionModHome<PCType,PTileType,PTDType,PType>;
        m_interactions[InteractionNames::work] = new InteractionModWork<PCType,PTileType,PTDType,PType>;
        m_interactions[InteractionNames::school] = new InteractionModSchool<PCType,PTileType,PTDType,PType>;
        m_interactions[InteractionNames::nborhood] = new InteractionModNborhood<PCType,PTileType,PTDType,PType>;
        m_interactions[InteractionNames::random] = new InteractionModRandom<PCType,PTileType, PTDType, PType>;

        m_hospital = std::make_unique<HospitalModel<PCType,PTileType,PTDType,PType>>();
    }

    m_h_parm.resize(m_num_diseases);
    m_d_parm.resize(m_num_diseases);

    for (int d = 0; d < m_num_diseases; d++) {
        m_h_parm[d] = new DiseaseParm{};
        m_d_parm[d] = (DiseaseParm*)amrex::The_Arena()->alloc(sizeof(DiseaseParm));

        m_h_parm[d]->readContact();
        // first read inputs common to all diseases
        m_h_parm[d]->readInputs("disease");
        // now read any disease-specific input, if available
        m_h_parm[d]->readInputs(std::string("disease_"+m_disease_names[d]));
        m_h_parm[d]->Initialize();

#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(m_d_parm[d], m_h_parm[d], sizeof(DiseaseParm));
#else
        std::memcpy(m_d_parm[d], m_h_parm[d], sizeof(DiseaseParm));
#endif
    }
    m_sc_period = m_d_parm[0]->incubation_length_mean + m_d_parm[0]->infectious_length_mean; /* close school relative to mean recovery time*/
}


/*! \brief Return bin pointer at a given mfi, tile and model name */
DenseBins<AgentContainer::PType>* AgentContainer::getBins (const std::pair<int,int>& a_idx,
                                                           const std::string& a_mod_name)
{
    BL_PROFILE("AgentContainer::getBins");
    if (a_mod_name == ExaEpi::InteractionNames::home) {
        return &m_bins_home[a_idx];
    } else if (    (a_mod_name == ExaEpi::InteractionNames::work)
                || (a_mod_name == ExaEpi::InteractionNames::school) ) {
        return &m_bins_work[a_idx];
    } else if (a_mod_name == ExaEpi::InteractionNames::nborhood) {
        if (m_at_work) { return &m_bins_work[a_idx]; }
        else           { return &m_bins_home[a_idx]; }
    } else if (a_mod_name == ExaEpi::InteractionNames::random) {
        return &m_bins_random[a_idx];
    } else {
        amrex::Abort("Invalid a_mod_name!");
        return nullptr;
    }
}

/*! \brief Send agents on a random walk around the neighborhood

    For each agent, set its position to a random one near its current position
*/
void AgentContainer::moveAgentsRandomWalk ()
{
    BL_PROFILE("AgentContainer::moveAgentsRandomWalk");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                ParticleType& p = pstruct[i];
                p.pos(0) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[0]);
                p.pos(1) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[1]);
            });
        }
    }
}

/*! \brief Move agents to work

    For each agent, set its position to the work community (IntIdx::work_i, IntIdx::work_j)
*/
void AgentContainer::moveAgentsToWork ()
{
    BL_PROFILE("AgentContainer::moveAgentsToWork");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                if (!isHospitalized(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    p.pos(0) = (work_i_ptr[ip] + 0.5_prt)*dx[0];
                    p.pos(1) = (work_j_ptr[ip] + 0.5_prt)*dx[1];
                }
            });
        }
    }

    m_at_work = true;

    Redistribute();
}

/*! \brief Move agents to home

    For each agent, set its position to the home community (IntIdx::home_i, IntIdx::home_j)
*/
void AgentContainer::moveAgentsToHome ()
{
    BL_PROFILE("AgentContainer::moveAgentsToHome");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                if (!isHospitalized(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    p.pos(0) = (home_i_ptr[ip] + 0.5_prt)*dx[0];
                    p.pos(1) = (home_j_ptr[ip] + 0.5_prt)*dx[1];
                }
            });
        }
    }

    m_at_work = false;

    Redistribute();
}

/*! \brief Move agents randomly

    For each agent, set its position to a random location with a probabilty of 0.01%
*/
void AgentContainer::moveRandomTravel ()
{
    BL_PROFILE("AgentContainer::moveRandomTravel");

    const Box& domain = Geom(0).Domain();
    int i_max = domain.length(0);
    int j_max = domain.length(1);
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();
            auto& soa   = ptile.GetStructOfArrays();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                if (!isHospitalized(i, ptd)) {
                    ParticleType& p = pstruct[i];
                    if (withdrawn_ptr[i] == 1) {return ;}
                    if (amrex::Random(engine) < 0.0001) {
                        random_travel_ptr[i] = i;
                        int i_random = int( amrex::Real(i_max)*amrex::Random(engine));
                        int j_random = int( amrex::Real(j_max)*amrex::Random(engine));
                        p.pos(0) = i_random;
                        p.pos(1) = j_random;
                    }
                }
            });
        }
    }
}

/*! \brief Return agents from random travel
*/
void AgentContainer::returnRandomTravel (const AgentContainer& on_travel_pc)
{
    BL_PROFILE("AgentContainer::returnRandomTravel");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);
        const auto& plev_travel = on_travel_pc.GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();

            const auto& ptile_travel = plev_travel.at(std::make_pair(gid, tid));
            const auto& aos_travel   = ptile_travel.GetArrayOfStructs();
            const size_t np_travel = aos_travel.numParticles();
            auto& soa_travel= ptile_travel.GetStructOfArrays();
            auto random_travel_ptr_travel = soa_travel.GetIntData(IntIdx::random_travel).data();

            int r_RT = RealIdx::nattribs;
            int n_disease = m_num_diseases;
            for (int d = 0; d < n_disease; d++) {
                auto prob_ptr        = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::prob).data();
                auto prob_ptr_travel = soa_travel.GetRealData(r_RT+r0(d)+RealIdxDisease::prob).data();

                amrex::ParallelFor( np_travel,
                    [=] AMREX_GPU_DEVICE (int i) noexcept
                    {
                        int dst_index = random_travel_ptr_travel[i];
                        prob_ptr[dst_index] += prob_ptr_travel[i];
                        AMREX_ALWAYS_ASSERT(random_travel_ptr[dst_index] = dst_index);
                        AMREX_ALWAYS_ASSERT(random_travel_ptr[dst_index] >= 0);
                        random_travel_ptr[dst_index] = -1;
                    });
            }
        }
    }
}

/*! \brief Updates disease status of each agent */
void AgentContainer::updateStatus ( MFPtrVec& a_disease_stats /*!< Community-wise disease stats tracker */)
{
    BL_PROFILE("AgentContainer::updateStatus");

    m_disease_status.updateAgents(*this, a_disease_stats);
    m_hospital->treatAgents(*this, a_disease_stats);

    // move hospitalized agents to their hospital location
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                if (isHospitalized(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    p.pos(0) = (hosp_i_ptr[ip] + 0.5_prt)*dx[0];
                    p.pos(1) = (hosp_j_ptr[ip] + 0.5_prt)*dx[1];
                }
            });
        }
    }
}

/*! \brief Start shelter-in-place */
void AgentContainer::shelterStart ()
{
    BL_PROFILE("AgentContainer::shelterStart");

    amrex::Print() << "Starting shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            auto shelter_compliance = m_shelter_compliance;
            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {
                if (amrex::Random(engine) < shelter_compliance) {
                    withdrawn_ptr[i] = 1;
                }
            });
        }
    }
}

/*! \brief Stop shelter-in-place */
void AgentContainer::shelterStop ()
{
    BL_PROFILE("AgentContainer::shelterStop");

    amrex::Print() << "Stopping shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                withdrawn_ptr[i] = 0;
            });
        }
    }
}

/*! \brief Infect agents based on their current status and the computed probability of infection.
    The infection probability is computed in AgentContainer::interactAgentsHomeWork() or
    AgentContainer::interactAgents() */
void AgentContainer::infectAgents ()
{
    BL_PROFILE("AgentContainer::infectAgents");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;
            int n_disease = m_num_diseases;

            for (int d = 0; d < n_disease; d++) {

                auto status_ptr = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();

                auto counter_ptr           = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();
                auto prob_ptr              = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::prob).data();
                auto incubation_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::incubation_period).data();
                auto infectious_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::infectious_period).data();
                auto symptomdev_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::symptomdev_period).data();

                auto* lparm = m_d_parm[d];

                amrex::ParallelForRNG( np,
                [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
                {
                    prob_ptr[i] = 1.0_rt - prob_ptr[i];
                    if ( status_ptr[i] == Status::never ||
                         status_ptr[i] == Status::susceptible ) {
                        if (amrex::Random(engine) < prob_ptr[i]) {
                            status_ptr[i] = Status::infected;
                            counter_ptr[i] = 0.0_rt;
                            incubation_period_ptr[i] = amrex::RandomNormal(lparm->latent_length_mean, lparm->latent_length_std, engine);
                            infectious_period_ptr[i] = amrex::RandomNormal(lparm->infectious_length_mean, lparm->infectious_length_std, engine);
                            symptomdev_period_ptr[i] = amrex::RandomNormal(lparm->incubation_length_mean, lparm->incubation_length_std, engine);
                            return;
                        }
                    }
                });
            }
        }
    }
}

/*! \brief Computes the number of agents with various #Status in each grid cell of the
    computational domain.

    Given a MultiFab with at least 5 x (number of diseases) components that is defined with
    the same box array and distribution mapping as this #AgentContainer, the MultiFab will
    contain (at the end of this function) the following *in each cell*:
    For each disease (d being the disease index):
    + component 5*d+0: total number of agents in this grid cell.
    + component 5*d+1: number of agents that have never been infected (#Status::never)
    + component 5*d+2: number of agents that are infected (#Status::infected)
    + component 5*d+3: number of agents that are immune (#Status::immune)
    + component 5*d+4: number of agents that are susceptible infected (#Status::susceptible)
*/
void AgentContainer::generateCellData (MultiFab& mf /*!< MultiFab with at least 5*m_num_diseases components */) const
{
    BL_PROFILE("AgentContainer::generateCellData");

    const int lev = 0;

    AMREX_ASSERT(OK());
    AMREX_ASSERT(numParticlesOutOfRange(*this, 0) == 0);

    const auto& geom = Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto domain = geom.Domain();
    int n_disease = m_num_diseases;

    ParticleToMesh(*this, mf, lev,
        [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                              int i,
                              Array4<Real> const& count)
        {
            auto p = ptd.m_aos[i];
            auto iv = getParticleCell(p, plo, dxi, domain);

            for (int d = 0; d < n_disease; d++) {
                int status = ptd.m_runtime_idata[i0(d)+IntIdxDisease::status][i];
                Gpu::Atomic::AddNoRet(&count(iv, 5*d+0), 1.0_rt);
                if (status != Status::dead) {
                    Gpu::Atomic::AddNoRet(&count(iv, 5*d+status+1), 1.0_rt);
                }
            }
        }, false);
}

/*! \brief Computes the total number of agents with each #Status

    Returns a vector with 5 components corresponding to each value of #Status; each element is
    the total number of agents at a step with the corresponding #Status (in that order).
*/
std::array<Long, 9> AgentContainer::getTotals (const int a_d /*!< disease index */) {
    BL_PROFILE("getTotals");
    amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    auto r = amrex::ParticleReduce<ReduceData<int,int,int,int,int,int,int,int,int>> (
                  *this, [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd, const int i) noexcept
                  -> amrex::GpuTuple<int,int,int,int,int,int,int,int,int>
              {
                  int s[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                  auto status = ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][i];

                  AMREX_ALWAYS_ASSERT(status >= 0);
                  AMREX_ALWAYS_ASSERT(status <= 4);

                  s[status] = 1;

                  if (status == Status::infected) {  // exposed
                      if (notInfectiousButInfected(i, ptd, a_d)) {
                          s[5] = 1;  // exposed, but not infectious
                      } else { // infectious
                          if (ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::symptomatic][i] == SymptomStatus::asymptomatic) {
                              s[6] = 1;  // asymptomatic and will remain so
                          }
                          else if (ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::symptomatic][i] == SymptomStatus::presymptomatic) {
                              s[7] = 1;  // asymptomatic but will develop symptoms
                          }
                          else if (ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::symptomatic][i] == SymptomStatus::symptomatic) {
                              s[8] = 1;  // Infectious and symptomatic
                          } else {
                              amrex::Abort("how did I get here?");
                          }
                      }
                  }
                  return {s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]};
              }, reduce_ops);

    std::array<Long, 9> counts = {amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r), amrex::get<3>(r),
                                  amrex::get<4>(r), amrex::get<5>(r), amrex::get<6>(r), amrex::get<7>(r),
                                  amrex::get<8>(r)};
    ParallelDescriptor::ReduceLongSum(&counts[0], 9, ParallelDescriptor::IOProcessorNumber());
    return counts;
}

/*! \brief Interaction and movement of agents during morning commute
 *
 * + Move agents to work
 * + Simulate interactions during morning commute (public transit/carpool/etc ?)
*/
void AgentContainer::morningCommute ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::morningCommute");
    //if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //    m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    //}
    moveAgentsToWork();
}

/*! \brief Interaction and movement of agents during evening commute
 *
 * + Simulate interactions during evening commute (public transit/carpool/etc ?)
 * + Simulate interactions at locations agents may stop by on their way home
 * + Move agents to home
*/
void AgentContainer::eveningCommute ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::eveningCommute");
    //if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //    m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    //}
    //if (haveInteractionModel(ExaEpi::InteractionNames::grocery_store)) {
    //    m_interactions[ExaEpi::InteractionNames::grocery_store]->interactAgents( *this, a_mask_behavior );
    //}
    moveAgentsToHome();
}

/*! \brief Interaction of agents during day time - work and school */
void AgentContainer::interactDay ( MultiFab& a_mask_behavior /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::interactDay");
    if (haveInteractionModel(ExaEpi::InteractionNames::work)) {
        m_interactions[ExaEpi::InteractionNames::work]->interactAgents( *this, a_mask_behavior );
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::school)) {
        m_interactions[ExaEpi::InteractionNames::school]->interactAgents( *this, a_mask_behavior );
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::nborhood)) {
        m_interactions[ExaEpi::InteractionNames::nborhood]->interactAgents( *this, a_mask_behavior );
    }

    m_hospital->interactAgents(*this, a_mask_behavior);
}

/*! \brief Interaction of agents during evening (after work) - social stuff */
void AgentContainer::interactEvening ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::interactEvening");
}

/*! \brief Interaction of agents during nighttime time - at home */
void AgentContainer::interactNight ( MultiFab& a_mask_behavior /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::interactNight");
    if (haveInteractionModel(ExaEpi::InteractionNames::home)) {
        m_interactions[ExaEpi::InteractionNames::home]->interactAgents( *this, a_mask_behavior );
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::nborhood)) {
        m_interactions[ExaEpi::InteractionNames::nborhood]->interactAgents( *this, a_mask_behavior );
    }
}

/*! \brief Interaction with agents on random travel */
void AgentContainer::interactRandomTravel ( MultiFab& a_mask_behavior, /*!< Masking behavior */
                                            AgentContainer& on_travel_pc /*< agents that are on random_travel */)
{
    BL_PROFILE("AgentContainer::interactNight");
    if (haveInteractionModel(ExaEpi::InteractionNames::random)) {
        m_interactions[ExaEpi::InteractionNames::random]->interactAgents( *this, a_mask_behavior, on_travel_pc);
    }
}

void AgentContainer::updateSchoolInfection(DemographicData& demo, iMultiFab& unit_mf, iMultiFab& comm_mf,iMultiFab& a_school_stats /*!< Community-wise school infection stats and status tracker */)
{
    BL_PROFILE("AgentContainer::updateSchoolInfo");

    struct SchoolDismissal
    {
        enum {
            ByCommunity = 0,   /*!< whether school is open or close */
            BySchool,   /*!< total infected student in community if school open */
            ByUnit  /*!< day count of school being closed */
        };
    };

    amrex::ParmParse pp("agent");
    std::string school_dismissal_option = "by_community";
    pp.query("school_dismissal_option", school_dismissal_option);
    int school_dismissal_flag = SchoolDismissal::ByCommunity;
    if (school_dismissal_option == "by_community"){school_dismissal_flag = SchoolDismissal::ByCommunity; }
    else if (school_dismissal_option == "by_school"){school_dismissal_flag = SchoolDismissal::BySchool; }
    else if (school_dismissal_option == "by_unit"){school_dismissal_flag = SchoolDismissal::ByUnit; }

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();

            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto school_ptr = soa.GetIntData(IntIdx::school).data();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto unit_arr = unit_mf[mfi].array();

            // int n_disease = m_num_diseases;
            // if (n_disease > 1) {
            //     throw std::runtime_error("Multiple diseases not supported for now for school dismissal - deactivate school dismissal flag");
            // }

            auto sc_infection_threshold = m_sc_infection_threshold;
            auto sc_period = m_sc_period;

            auto ss_arr = a_school_stats[mfi].array();
            auto infection_reset_arr = reset_school_infection[mfi].array();
            const auto& sc_arr = student_counts[mfi].array();

            const Box& bx = mfi.tilebox();

            struct SchoolStats
            {
                enum {
                    SchoolDismissal = 0,   /*!< whether school is open or close */
                    SchoolInfectionCount,   /*!< total infected student in community if school open */
                    SchoolStatusDayCount  /*!< day count of school being closed */
                };
            };
            int nattr = SchoolType::nattribs;

            /* Infection Counts at a given day */
            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    /* Reset count each day */
                    if (school_dismissal_flag == SchoolDismissal::ByCommunity || school_dismissal_flag == SchoolDismissal::ByUnit){
                        if (ss_arr(i, j, k, nattr*SchoolStats::SchoolDismissal) == 0 && ss_arr(i, j, k, nattr*SchoolStats::SchoolStatusDayCount) == 0){
                            infection_reset_arr(i,j,k,0) = -1 * ss_arr(i, j, k, nattr*SchoolStats::SchoolInfectionCount);
                        }
                        else if (ss_arr(i, j, k, nattr*SchoolStats::SchoolDismissal) == 1){infection_reset_arr(i,j,k,0) = 0; }

                        for (int ii = 0; ii < 5; ++ii){
                            ss_arr(i, j, k, (ii+1)+nattr*SchoolStats::SchoolInfectionCount) = 0;
                        }
                        ss_arr(i, j, k, nattr*SchoolStats::SchoolInfectionCount) = infection_reset_arr(i,j,k,0);

                    }
                    else if (school_dismissal_flag == SchoolDismissal::BySchool){
                        for (int ii = 1; ii < 5; ++ii){
                            if (ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolDismissal) == 0 && ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount) == 0){
                                infection_reset_arr(i,j,k,ii) = -1 * ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolInfectionCount);
                            }
                            else if (ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolDismissal) == 1){infection_reset_arr(i,j,k,ii) = 0; }

                            ss_arr(i, j, k, nattr*SchoolStats::SchoolInfectionCount) = 0;
                            ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolInfectionCount) = infection_reset_arr(i,j,k,ii);
                        }
                    }

                    // Count infections
                    for (int p = 0; p < np; ++p) {
                        if (home_i_ptr[p] == i && home_j_ptr[p] == j && age_group_ptr[p] == 1) {
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(school_ptr[p] >= 0, "School_ptr can't be negative when school's open");
                            if (school_ptr[p]) {
                                if (withdrawn_ptr[p] == 1 || hosp_i_ptr[p] > -1) {
                                    amrex::Gpu::Atomic::Add(&ss_arr(i, j, k, nattr*SchoolStats::SchoolInfectionCount), 1);

                                    if (school_ptr[p] == SchoolType::high   || school_ptr[p] == -1*SchoolType::high){
                                        amrex::Gpu::Atomic::Add(&ss_arr(i, j, k, 1+nattr*SchoolStats::SchoolInfectionCount), 1);
                                    }
                                    if (school_ptr[p] == SchoolType::middle || school_ptr[p] == -1*SchoolType::middle){
                                        amrex::Gpu::Atomic::Add(&ss_arr(i, j, k, 2+nattr*SchoolStats::SchoolInfectionCount), 1);
                                    }
                                    if (school_ptr[p] == SchoolType::elem_3 || school_ptr[p] == -1*SchoolType::elem_3){
                                        amrex::Gpu::Atomic::Add(&ss_arr(i, j, k, 3+nattr*SchoolStats::SchoolInfectionCount), 1);
                                    }
                                    if (school_ptr[p] == SchoolType::elem_4 || school_ptr[p] == -1*SchoolType::elem_4){
                                        amrex::Gpu::Atomic::Add(&ss_arr(i, j, k, 4+nattr*SchoolStats::SchoolInfectionCount), 1);
                                    }

                                }
                            }
                        }
                    }
                });

                Gpu::synchronize();

            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    // if (school_dismissal_flag == SchoolDismissal::ByCommunity || school_dismissal_flag == SchoolDismissal::ByUnit)
                    int start_dismis = 0;
                    int stop_dismiss = 1;
                    if (school_dismissal_flag == SchoolDismissal::BySchool){
                        start_dismis = 1;
                        stop_dismiss = 5;
                    }

                    for (int ii = start_dismis; ii < stop_dismiss; ++ii)
                    {
                        if (ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolDismissal) == 0) {
                            // Check if the school should be closed
                            int student_total;
                            if (ii == 0){
                                student_total = sc_arr(i, j, k, SchoolType::elem_3)
                                        + sc_arr(i, j, k, SchoolType::elem_4)
                                        + sc_arr(i, j, k, SchoolType::middle)
                                        + sc_arr(i, j, k, SchoolType::high); // handle playgroup later
                            }
                            else{ student_total = sc_arr(i, j, k, ii);}

                            amrex::Real thresh; // threshold can be either fixed # of student, or a proportion of total students
                            if (sc_infection_threshold >= 1){ thresh =  sc_infection_threshold;}
                            else {thresh = sc_infection_threshold * student_total;}

                            if (ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolInfectionCount) >= thresh) {
                                ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolDismissal) = 1;
                                ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount) = 1;
                                if (unit_arr(i,j,k) == 164 ) {
#ifdef AMREX_USE_CUDA
                                    printf("School %d at (%d, %d, %d) is now closed %d. \nInfection number: MultiFab = %d, Day = %d\n", ii,
                                        i, j, k,
                                        ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolDismissal),
                                        ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolInfectionCount),
                                        ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount));
#endif
                                }
                            }
                            else {
                                amrex::Gpu::Atomic::Add(&ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount), 1);

                                if (unit_arr(i,j,k) == 164) {
#ifdef AMREX_USE_CUDA
                                    printf("School %d at (%d, %d, %d) is currenly opened %d. Infection number: MultiFab = %d, Day = %d\n", ii,
                                        i, j, k,
                                        ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolDismissal),
                                        ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolInfectionCount),
                                        ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount));
#endif
                                }

                            }
                        } else { // School is closed
                            if (ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount) > sc_period) {
                                ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolDismissal) = 0;
                                ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount) = 0;

                                if (unit_arr(i,j,k) == 164) {
#ifdef AMREX_USE_CUDA
                                printf("School %d at (%d, %d, %d) has opened %d. Infection number: MultiFab = %d, Day = %d\n", ii,
                                    i, j, k,
                                    ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolDismissal),
                                    ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolInfectionCount),
                                    ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount));
#endif

                                }
                            } else {
                                amrex::Gpu::Atomic::Add(&ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount), 1);
                                if (unit_arr(i,j,k) == 164) {
#ifdef AMREX_USE_CUDA
                                    printf("School %d at (%d, %d, %d) is currently closed %d. Infection number: MultiFab = %d, Day = %d\n", ii,
                                        i, j, k,
                                        ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolDismissal),
                                        ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolInfectionCount),
                                        ss_arr(i, j, k, ii+nattr*SchoolStats::SchoolStatusDayCount));
#endif
                                }
                            }
                        }
                    }
                });
            Gpu::synchronize();

            if (school_dismissal_flag == SchoolDismissal::ByUnit)
            {
                auto Start = demo.Start_d.data();
                int Ncommunity = demo.Ncommunity;
                int Nx = (int) std::floor(std::sqrt((double) Ncommunity));
                int Ny = Nx;

                // Adjust Nx
                while (Nx * Ny < Ncommunity) {
                    ++Nx;
                }

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    if (ss_arr(i, j, k, nattr*SchoolStats::SchoolDismissal) == 1 && ss_arr(i, j, k, nattr*SchoolStats::SchoolStatusDayCount) == 1) {
                        int to = unit_arr(i,j,k);
                        int start_comm = Start[to];
                        int stop_comm = Start[to+1];

                        for (int comm_close = start_comm; comm_close < stop_comm; ++comm_close){
                            int close_k = comm_close / (Nx * Ny);
                            int close_j = (comm_close % (Nx * Ny)) / Nx;
                            int close_i = comm_close % Nx;
                            ss_arr(close_i, close_j, close_k, nattr*SchoolStats::SchoolDismissal) = 1;
                            ss_arr(close_i, close_j, close_k, nattr*SchoolStats::SchoolStatusDayCount) = 1;

                            if (unit_arr(close_i,close_j,close_k) == 164 ) {
#ifdef AMREX_USE_CUDA
                                printf("School %d at (%d, %d, %d) should close %d. UNIT: %d,Comm: %d \nInfection number: MultiFab = %d, Day = %d\n", 0,
                                    close_i, close_j, close_k,
                                    ss_arr(close_i, close_j, close_k, nattr*SchoolStats::SchoolDismissal),
                                    to, comm_close,
                                    ss_arr(close_i, close_j, close_k, nattr*SchoolStats::SchoolInfectionCount),
                                    ss_arr(close_i, close_j, close_k, nattr*SchoolStats::SchoolStatusDayCount));
#endif
                            }
                        }
                    }
                    else if (ss_arr(i, j, k, nattr*SchoolStats::SchoolDismissal) == 0 && ss_arr(i, j, k, nattr*SchoolStats::SchoolStatusDayCount) == 0){
                        int to = unit_arr(i,j,k);
                        int start_comm = Start[to];
                        int stop_comm = Start[to+1];

                        for (int comm_open = start_comm; comm_open < stop_comm; ++comm_open){
                            int open_k = comm_open / (Nx * Ny);
                            int open_j = (comm_open % (Nx * Ny)) / Nx;
                            int open_i = comm_open % Nx;
                            ss_arr(open_i, open_j, open_k, nattr*SchoolStats::SchoolDismissal) = 0;
                            ss_arr(open_i, open_j, open_k, nattr*SchoolStats::SchoolStatusDayCount) = 0;

                            if (unit_arr(open_i,open_j,open_k) == 164 ) {
#ifdef AMREX_USE_CUDA
                                printf("School %d at (%d, %d, %d) should open %d. UNIT: %d,Comm: %d \nInfection number: MultiFab = %d, Day = %d\n", 0,
                                    open_i, open_j, open_k,
                                    ss_arr(open_i, open_j, open_k, nattr*SchoolStats::SchoolDismissal),
                                    to, comm_open,
                                    ss_arr(open_i, open_j, open_k, nattr*SchoolStats::SchoolInfectionCount),
                                    ss_arr(open_i, open_j, open_k, nattr*SchoolStats::SchoolStatusDayCount));
#endif
                            }
                        }
                    }
                });
            Gpu::synchronize();
            }

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int p) noexcept
            {

                if (age_group_ptr[p] >= 1 && school_ptr[p]){ // teachers and student

                    int school_type = 0;
                    if (school_dismissal_flag == SchoolDismissal::BySchool){
                        school_type = (school_ptr[p] < 0) ? -school_ptr[p] : school_ptr[p];
                    }

                    if (ss_arr(home_i_ptr[p], home_j_ptr[p], 0, school_type+nattr*SchoolStats::SchoolDismissal) == 0
                     && ss_arr(home_i_ptr[p], home_j_ptr[p], 0, school_type+nattr*SchoolStats::SchoolStatusDayCount) == 0){
                        if (school_ptr[p] < 0) { school_ptr[p] *= -1; }
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(school_ptr[p] >= 0, "School_ptr can't be positive when school's is open");
                    }
                    else if (ss_arr(home_i_ptr[p], home_j_ptr[p], 0, school_type+nattr*SchoolStats::SchoolDismissal) == 1
                          && ss_arr(home_i_ptr[p], home_j_ptr[p], 0, school_type+nattr*SchoolStats::SchoolStatusDayCount) == 1){
                        if (school_ptr[p] > 0) { school_ptr[p] *= -1; }
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(school_ptr[p] <= 0, "School_ptr can't be positive when school's is close");

                    }
                }
            });
            Gpu::synchronize();
        }
    }
}


// void AgentContainer::printSchoolInfection(iMultiFab& unit_mf, iMultiFab& a_school_stats) const {
//     int n_disease = m_num_diseases;
//     int total_std_fab = 0;
//     int total_infec_fab = 0;
//     int total_infec_sim = 0;
//     int total_std_sim = 0;

//     // if (n_disease > 1) {
//     //     throw std::runtime_error("Multiple diseases not supported");
//     // }

//     for (int lev = 0; lev <= finestLevel(); ++lev) {
//         auto& plev = GetParticles(lev);

// #ifdef AMREX_USE_OMP
// #pragma omp parallel if (Gpu::notInLaunchRegion())
// #endif
//         {
//             int local_total_infec_fab = 0;
//             int local_total_infec_sim = 0;
//             int local_total_std_fab = 0;
//             int local_total_std_sim = 0;

//             for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

//                 int gid = mfi.index();
//                 int tid = mfi.LocalTileIndex();
//                 auto& ptile = plev.at(std::make_pair(gid, tid));
//                 auto& soa = ptile.GetStructOfArrays();
//                 const auto np = ptile.numParticles();

//                 auto timer_ptr = soa.GetRealData(RealIdx::treatment_timer).data();
//                 auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
//                 auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
//                 auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
//                 auto school_ptr = soa.GetIntData(IntIdx::school).data();
//                 const auto& student_counts_arr = student_counts[mfi].array();
//                 auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
//                 auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
//                 auto ss_arr = a_school_stats[mfi].array();
//                 const amrex::Box& bx = mfi.tilebox();
//                 auto unit_arr = unit_mf[mfi].array();

//                 struct SchoolStats
//                 {
//                     enum {
//                         SchoolDismissal = 0,   /*!< whether school is open or close */
//                         SchoolInfectionCount,   /*!< total infected student in community if school open */
//                         SchoolStatusDayCount  /*!< day count of school being closed */
//                     };
//                 };
//                 int nattr = SchoolType::nattribs;

//                 for (amrex::IntVect iv = bx.smallEnd(); iv <= bx.bigEnd(); bx.next(iv)) {
//                     int i = iv[0];
//                     int j = iv[1];
//                     int k = 0; // Assuming 2D; use iv[2] for 3D
//                     int count_infec = 0;
//                     int infect_high = 0;
//                     int infect_middle = 0;
//                     int infect_elem3 = 0;
//                     int infect_elem4 = 0;
//                     int infect_high_fab = 0;
//                     int infect_middle_fab = 0;
//                     int infect_elem3_fab = 0;
//                     int infect_elem4_fab = 0;
//                     int count_std = 0;
//                     int fab_total = student_counts_arr(i, j, k, SchoolType::elem_3)
//                                     + student_counts_arr(i, j, k, SchoolType::elem_4)
//                                     + student_counts_arr(i, j, k, SchoolType::middle)
//                                     + student_counts_arr(i, j, k, SchoolType::high);

//                     for (int p = 0; p < np; ++p) {
//                         if (home_i_ptr[p] == i && home_j_ptr[p] == j && age_group_ptr[p] == 1 && school_ptr[p]) {
//                             ++count_std;
//                             if (withdrawn_ptr[p] || hosp_i_ptr[p] > -1){
//                                 ++count_infec;
//                                 if (school_ptr[p] == SchoolType::high   || school_ptr[p] == -1*SchoolType::high){++infect_high;}
//                                 if (school_ptr[p] == SchoolType::middle || school_ptr[p] == -1*SchoolType::middle){++infect_middle;}
//                                 if (school_ptr[p] == SchoolType::elem_3 || school_ptr[p] == -1*SchoolType::elem_3){++infect_elem3;}
//                                 if (school_ptr[p] == SchoolType::elem_4 || school_ptr[p] == -1*SchoolType::elem_4){++infect_elem4;}
//                             }
//                             for (int d = 0; d < n_disease; d++) {
//                                 auto status_ptr = soa.GetIntData(IntIdx::nattribs+i0(d)+IntIdxDisease::status).data();
//                                 if (unit_arr(i,j,k) == 164 && status_ptr[p] == Status::dead){std::cout << " Agent " << p << "is Dead at comm. ("
//                                                                                                                     << i << " ," << j << " ," << k << ")"<< std::endl;}
//                             }
//                         }
//                     }
//                     local_total_std_fab += fab_total; // student count
//                     local_total_infec_sim += count_infec;
//                     local_total_infec_fab += ss_arr(iv, nattr*SchoolStats::SchoolInfectionCount);
//                     local_total_std_sim += count_std;

//                     infect_high_fab   += ss_arr(iv, 1+nattr*SchoolStats::SchoolInfectionCount);
//                     infect_middle_fab += ss_arr(iv, 2+nattr*SchoolStats::SchoolInfectionCount);
//                     infect_elem3_fab  += ss_arr(iv, 3+nattr*SchoolStats::SchoolInfectionCount);
//                     infect_elem4_fab  += ss_arr(iv, 4+nattr*SchoolStats::SchoolInfectionCount);

//                     if (unit_arr(i,j,k) == 164){
//                         std::cout << "School Infection number at ("
//                                     << i << ", " << j << ", " << k << "): MultiFab = "
//                                     << ss_arr(i, j, k, nattr*SchoolStats::SchoolInfectionCount) << ", Sim = "
//                                     << count_infec << "\n"
//                                     << "  SIM Infected High School: " << infect_high << "\n"
//                                     << "  FAB Infected High School: " << ss_arr(i, j, k, 1+nattr*SchoolStats::SchoolInfectionCount) << "\n"
//                                     << "  SIM Infected Middle School: " << infect_middle << "\n"
//                                     << "  FAB Infected Middle School: " << ss_arr(i, j, k, 2+nattr*SchoolStats::SchoolInfectionCount) << "\n"
//                                     << "  SIM Infected Elementary School Neighborhood 1: " << infect_elem3 << "\n"
//                                     << "  FAB Infected Elementary School Neighborhood 1: " << ss_arr(i, j, k, 3+nattr*SchoolStats::SchoolInfectionCount) << "\n"
//                                     << "  SIM Infected Elementary School Neighborhood 2: " << infect_elem4 << "\n"
//                                     << "  FAB Infected Elementary School Neighborhood 2: " << ss_arr(i, j, k, 4+nattr*SchoolStats::SchoolInfectionCount) << std::endl;
//                         // std::cout << "School student numbers at ("
//                         //         << i << ", " << j << ", " << k << "):\n"
//                         //         << "  Total: " << student_counts_arr(i, j, k, SchoolType::total) << "\n"
//                         //         << "  High School: " << student_counts_arr(i, j, k, SchoolType::high) << "\n"
//                         //         << "  Middle School: " << student_counts_arr(i, j, k, SchoolType::middle) << "\n"
//                         //         << "  Elementary School Neighborhood 1: " << student_counts_arr(i, j, k, SchoolType::elem_3) << "\n"
//                         //         << "  Elementary School Neighborhood 2: " << student_counts_arr(i, j, k, SchoolType::elem_4) << "\n"
//                         //         << "  Day Care: " << student_counts_arr(i, j, k, SchoolType::day_care) << "\n"
//                         //         << "  Total Sim: " << count_std << "\n"
//                         //         << "  Without Daycare (Fab Total): " << fab_total << std::endl;
//                     }
//                 }

//             }

// #ifdef AMREX_USE_OMP
// #pragma omp atomic
// #endif
//             total_infec_fab += local_total_infec_fab;
// #ifdef AMREX_USE_OMP
// #pragma omp atomic
// #endif
//             total_infec_sim += local_total_infec_sim;
// #ifdef AMREX_USE_OMP
// #pragma omp atomic
// #endif
//             total_std_fab += local_total_std_fab;
// #ifdef AMREX_USE_OMP
// #pragma omp atomic
// #endif
//             total_std_sim += local_total_std_sim;
//         }
//     }

//     // Ensure that this block is executed only once, ideally outside of any parallel regions
//     // or placed in a section of the code that is guaranteed to execute after all computations
//     // are completed.
//     Print() << "Total infection count from MultiFab: " << total_infec_fab << std::endl;
//     Print() << "Total infection count from Simulation: " << total_infec_sim << std::endl;
//     Print() << "Total student count from Student mf: " << total_std_fab << std::endl;
// }

