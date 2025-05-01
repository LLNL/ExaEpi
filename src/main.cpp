/*! @file main.cpp
    \brief **Main**: Contains main() and runAgent()
*/

#include <chrono>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_iMultiFab.H>

#include "AgentContainer.H"
#include "AirTravelFlow.H"
#include "CaseData.H"
#include "DemographicData.H"
#include "IO.H"
#include "InitializeInfections.H"
#include "UrbanPopData.H"
#include "Utils.H"

#include "version.h"

using namespace amrex;
using namespace ExaEpi;

void runAgent();

/*! \brief Set ExaEpi-specific defaults for memory-management and tiling */
void overrideAmrexDefaults () {
    amrex::ParmParse pp("amrex");
    // ExaEpi should never require mananaged memory in the Arena
    bool the_arena_is_managed = true;
    pp.queryAdd("the_arena_is_managed", the_arena_is_managed);

    bool use_comms_arena = true;
    pp.queryAdd("use_comms_arena", use_comms_arena);

    amrex::ParmParse pp2("particles");
    // enable for CPUs, disable for GPUs
    bool do_tiling = TilingIfNotGPU();
    pp2.queryAdd("do_tiling", do_tiling);
}

/*! \brief Main function: initializes AMReX, calls runAgent(), finalizes AMReX */
int main (int argc, /*!< Number of command line arguments */
          char* argv[] /*!< Command line arguments */) {
    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, overrideAmrexDefaults);

    Print() << "ExaEpi version " << EXAEPI_VERSION << " (built on " << __DATE__ << ")\n";

    runAgent();

    amrex::Finalize();
}

/*! \brief Run agent-based simulation:

    \b Initialization
    + Read test parameters (#ExaEpi::TestParams) from command line input file
    + If initialization type (#ExaEpi::TestParams::ic_type) is ExaEpi::ICType::Census,
      + Read #DemographicData from #ExaEpi::TestParams::census_filename
        (see DemographicData::initFromFile)
      + Read #CaseData from #ExaEpi::TestParams::case_filename
        (see CaseData::initFromFile)
    + Get computational domain from ExaEpi::Utils::getGeometry. Each grid cell corresponds to
      a community.
    + Create box arrays and distribution mapping based on #ExaEpi::TestParams::max_box_size.
    + Initialize the following MultiFabs:
      + Number of residents: 6 components - number of residents in age groups under-5, 5-17,
        18-29, 30-64, 65+, total.
      + Unit number of the community at each grid cell (1 component).
      + FIPS code of the community at each grid cell (2 components - FIPS code, census tract ID).
      + Community number of the community at each grid cell.
      + Disease statistics with 4 components (hospitalization, ICU, ventilator, deaths)
      + Masking behavior
    + Initialize agents (AgentContainer::initAgentsCensus).
      If ExaEpi::TestParams::ic_type is ExaEpi::ICType::Census, then
      + Read worker flow (ExaEpi::Initialization::readWorkerflow)
      + Initialize cases (ExaEpi::Initialization::setInitialCases)


    \b Evolution
    At each step from 0 to #ExaEpi::TestParams::nsteps-1:
    + IO:
      + if the current step number is a multiple of #ExaEpi::TestParams::plot_int, then write
        out plot file - see ExaEpi::IO::writePlotFile()
      + if current step number is a multiple of #ExaEpi::TestParams::aggregated_diag_int, then write
        out aggregated diagnostic data - see ExaEpi::IO::writeFIPSData().
    + Agents behavior:
      + Update agent #Status based on their age, number of days since infection, hospitalization,
        etc. - see AgentContainer::updateStatus().
      + Move agents to work - see AgentContainer::moveAgentsToWork().
      + Let agents interact at work - see AgentContainer::interactAgentsHomeWork().
      + Move agents to home - see AgentContainer::moveAgentsToHome().
      + Let agents interact at home - see AgentContainer::interactAgentsHomeWork().
      + Infect agents based on their movements during the day - see AgentContainer::infectAgents().
    + Get disease statistics counts - see AgentContainer::printTotals() - and update the
      peak number of infections and cumulative deaths.

    \b Finalize
    + Report peak infections, day of peak infections, and cumulative deaths.
    + Write out final plot file - see ExaEpi::IO::writePlotFile()
    + Write out final aggregated diagnostic data - see ExaEpi::IO::writeFIPSData().
*/
void runAgent () {
    BL_PROFILE("runAgent");
    TestParams params;
    ExaEpi::Utils::getTestParams(params, "agent");

    amrex::Print() << "Tracking " << params.num_diseases << " diseases:\n";
    for (int d = 0; d < params.num_diseases; d++) {
        amrex::Print() << "    " << params.disease_names[d] << "\n";
    }

    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
    CensusData censusData;
    UrbanPopData urbanPopData;

    if (params.ic_type == ICType::Census) {
        censusData.init(params, geom, ba, dm);
    } else if (params.ic_type == ICType::UrbanPop) {
        urbanPopData.init(params, geom, ba, dm);
    }

    AirTravelFlow air;
    if (params.air_travel_int > 0) {
        air.readAirports(params.airports_filename, censusData.demo);
        air.readAirTravelFlow(params.air_traffic_filename);
        air.computeTravelProbs(censusData.demo);
    }

    // The default output filename is:
    // output.dat for a single disease
    // output_<disease_name>.dat for multiple diseases
    std::vector<std::string> output_filename;
    output_filename.resize(params.num_diseases);
    if (params.num_diseases == 1) {
        output_filename[0] = "output.dat";
    } else {
        for (int d = 0; d < params.num_diseases; d++) {
            output_filename[d] = "output_" + params.disease_names[d] + ".dat";
        }
    }
    ParmParse pp("diag");
    pp.queryarr("output_filename", output_filename, 0, params.num_diseases);

    for (int d = 0; d < params.num_diseases; d++) {
        if (ParallelDescriptor::IOProcessor()) {
            std::ofstream File;
            File.open(output_filename[d].c_str(), std::ios::out | std::ios::trunc);

            if (!File.good()) { amrex::FileOpenFailed(output_filename[d]); }

            File << std::setw(5) << "Day";
            File << std::setw(12) << "Susceptible";
            File << std::setw(12) << "Infected";
            File << std::setw(12) << "Recovered";
            File << std::setw(12) << "Deaths";
            File << std::setw(15) << "Hospitalized";
            File << std::setw(15) << "ICU";
            File << std::setw(12) << "Ventilated";
            File << std::setw(12) << "Exposed";
            File << std::setw(15) << "Asymptomatic";
            File << std::setw(15) << "Presymptomatic";
            File << std::setw(15) << "Symptomatic\n";

            File.flush();

            File.close();

            if (!File.good()) { amrex::Abort("problem writing output file"); }
        }
    }

    amrex::Vector<std::unique_ptr<MultiFab>> disease_stats;
    disease_stats.resize(params.num_diseases);
    for (int d = 0; d < params.num_diseases; d++) {
        disease_stats[d] = std::make_unique<MultiFab>(ba, dm, 5, 0);
        disease_stats[d]->setVal(0);
    }

    MultiFab mask_behavior(ba, dm, 1, 0);
    mask_behavior.setVal(1);

    AgentContainer pc(geom, dm, ba, params.num_diseases, params.disease_names, params.fast, params.ic_type);
    bool stable_redistribute = !params.fast;
    pc.setStableRedistribute(stable_redistribute);
    pc.setTileSize(censusData.unit_mf.mfiter_tile_size);

    {
        BL_PROFILE_REGION("Initialization");
        if (params.ic_type == ICType::Census) {
            censusData.initAgents(pc, params.nborhood_size);
            censusData.readWorkerflow(pc, params.workerflow_filename, params.workgroup_size);
        } else if (params.ic_type == ICType::UrbanPop) {
            urbanPopData.initAgents(pc, params);
        } else {
            Abort("Unimplemented ic_type");
        }

        for (int d = 0; d < params.num_diseases; d++) {
            auto disease_params = pc.getDiseaseParameters_h(d);
            if (disease_params->initial_case_type == CaseTypes::file) {
                CaseData cases;
                cases.initFromFile(disease_params->disease_name, std::string(disease_params->case_filename));
                setInitialCasesFromFile(pc, cases, disease_params->disease_name, d,
                                        (params.ic_type == ICType::Census ? censusData.demo.FIPS : urbanPopData.FIPS_codes),
                                        (params.ic_type == ICType::Census ? censusData.demo.Start
                                                                          : urbanPopData.fips_community_start),
                                        (params.ic_type == ICType::Census ? censusData.comm_mf : urbanPopData.community_mf),
                                        params.fast);
            } else {
                setInitialCasesRandom(pc, disease_params->num_initial_cases, disease_params->disease_name, d,
                                      (params.ic_type == ICType::Census ? censusData.demo.Start
                                                                        : urbanPopData.fips_community_start),
                                      (params.ic_type == ICType::Census ? censusData.comm_mf : urbanPopData.community_mf),
                                      params.fast);
            }
        }

        pc.printStudentTeacherCounts();
        pc.printAgeGroupCounts();

        if (params.ic_type == ICType::Census && params.air_travel_int > 0) {
            pc.setAirTravel(censusData.unit_mf, air, censusData.demo);
        }
    }

// #define DUMP_INITIAL_AGENTS_ASCII
#ifdef DUMP_INITIAL_AGENTS_ASCII
    string agents_fname = std::string("agents.") + (params.ic_type == ICType::UrbanPop ? "urbanpop" : "census") + ".csv";
    pc.WriteAsciiFile(agents_fname);
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream agents_f(agents_fname, std::ios_base::app);
        agents_f << "#posx posy id cpu " << "treatment_timer " << "disease_counter " << "prob " << "incubation_period "
                 << "infectious_period " << "symptomdev_period " << "age_group " << "family " << "home_i " << "home_j "
                 << "work_i " << "work_j " << "hosp_i " << "hosp_j " << "nborhood " << "school " << "naics " << "workgroup "
                 << "work_nborhood " << "withdrawn " << "random_travel " << "status " << "symptomatic\n";
        agents_f.close();
    }
#endif

    std::vector<int> step_of_peak(params.num_diseases, 0);
    std::vector<Long> num_infected_peak(params.num_diseases, 0);
    std::vector<Long> cumulative_deaths(params.num_diseases, 0);

    // Store cumulative deaths workers, teachers, non workers
    std::vector<Long> num_infected_peak_work(params.num_diseases, 0);
    std::vector<Long> cumulative_deaths_work(params.num_diseases, 0);
    std::vector<Long> num_infected_peak_teachers(params.num_diseases, 0);
    std::vector<Long> cumulative_deaths_teachers(params.num_diseases, 0);
    std::vector<Long> num_infected_peak_nonwork(params.num_diseases, 0);
    std::vector<Long> cumulative_deaths_nonwork(params.num_diseases, 0);

    std::vector<Long> num_infected_peak_student(params.num_diseases, 0);
    std::vector<Long> cumulative_deaths_student(params.num_diseases, 0);

    // Store cumulative deaths per age group
    std::vector<std::array<Long, AgeGroups::total>> num_infected_peak_ag(params.num_diseases, {0});
    std::vector<std::array<Long, AgeGroups::total>> cumulative_deaths_ag(params.num_diseases, {0});

    // Store cumulative deaths student per school type
    std::vector<std::array<Long, SchoolType::total>> num_infected_peak_stud_school(params.num_diseases, {0});
    std::vector<std::array<Long, SchoolType::total>> cumulative_deaths_stud_school(params.num_diseases, {0});

    // Store cumulative deaths teachers per school type
    std::vector<std::array<Long, SchoolType::total>> num_infected_peak_teacher_school(params.num_diseases, {0});
    std::vector<std::array<Long, SchoolType::total>> cumulative_deaths_teacher_school(params.num_diseases, {0});

    for (int d = 0; d < params.num_diseases; d++) {
        auto counts = pc.getTotals(d);
        // Update peak infections
        if (counts[1] > num_infected_peak[d]) {
            num_infected_peak[d] = counts[1];
            step_of_peak[d] = 0;
        }
        cumulative_deaths[d] = counts[4];

        auto counts_work = pc.getTotalsWorkers(d);
        if (counts_work[1] > num_infected_peak_work[d]) {
            num_infected_peak_work[d] = counts_work[1];
        }
        cumulative_deaths_work[d] = counts_work[4];

        auto counts_teach = pc.getTotalsTeachers(d);
        if (counts_teach[1] > num_infected_peak_teachers[d]) {
            num_infected_peak_teachers[d] = counts_teach[1];
        }
        cumulative_deaths_teachers[d] = counts_teach[4];

        auto counts_nonwork = pc.getTotalsNonWorkers(d);
        if (counts_nonwork[1] > num_infected_peak_nonwork[d]) {
            num_infected_peak_nonwork[d] = counts_nonwork[1];
        }
        cumulative_deaths_nonwork[d] = counts_nonwork[4];

        auto counts_student = pc.getTotalsStudent(d);
        if (counts_student[1] > num_infected_peak_student[d]) {
            num_infected_peak_student[d] = counts_student[1];
        }
        cumulative_deaths_student[d] = counts_student[4];

        // Loop over all age groups
        for (int ag = 0; ag < AgeGroups::total; ag++) {
            auto counts_ag = pc.getTotalsAgeGroup(d, ag);
            if (counts_ag[1] > num_infected_peak_ag[d][ag]) {
                num_infected_peak_ag[d][ag] = counts_ag[1];
            }
            cumulative_deaths_ag[d][ag] = counts_ag[4];
        }
        // Loop over all school types
        for (int sch = 0; sch < SchoolType::total; sch++) {
            auto counts_std_sch = pc.getTotalsSchoolStudent(d, sch);
            if (counts_std_sch[1] > num_infected_peak_stud_school[d][sch]) {
                num_infected_peak_stud_school[d][sch] = counts_std_sch[1];
            }
            cumulative_deaths_stud_school[d][sch] = counts_std_sch[4];
        }
        for (int sch = 1; sch < SchoolType::total; sch++) {
            auto count_teach_sch = pc.getTotalsTeachersPerSchool(d,sch);
            if (count_teach_sch[1] > num_infected_peak_teacher_school[d][sch]) {
                num_infected_peak_teacher_school[d][sch] = count_teach_sch[1];
            }
            cumulative_deaths_teacher_school[d][sch] = count_teach_sch[4];
        }
    }


    amrex::Real cur_time = 0;

    Vector<Long> num_infected(params.num_diseases, 0);
    Vector<Long> cum_num_infected(params.num_diseases, 0);

    // Store num infected and cumulative infections
    Vector<Long> num_infected_work(params.num_diseases, 0);
    Vector<Long> cum_num_infected_work(params.num_diseases, 0);
    Vector<Long> num_infected_teachers(params.num_diseases, 0);
    Vector<Long> cum_num_infected_teachers(params.num_diseases, 0);
    Vector<Long> num_infected_nonwork(params.num_diseases, 0);
    Vector<Long> cum_num_infected_nowork(params.num_diseases, 0);

    Vector<Long> num_infected_student(params.num_diseases, 0);
    Vector<Long> cum_num_infected_student(params.num_diseases, 0);

    // Store num infected and cumulative infections per age group
    std::vector<std::array<Long, AgeGroups::total>> cum_num_infected_ag(params.num_diseases, {0});
    std::vector<std::array<Long, AgeGroups::total>> num_infected_ag(params.num_diseases, {0});

    // Store num infected and cumulative infections student per school type
    std::vector<std::array<Long, SchoolType::total>> cum_num_infected_stud_sch(params.num_diseases, {0});
    std::vector<std::array<Long, SchoolType::total>> num_infected_stud_sch(params.num_diseases, {0});

    // Store num infected and cumulative infections teachers per school type
    std::vector<std::array<Long, SchoolType::total>> cum_num_infected_teach_sch(params.num_diseases, {0});
    std::vector<std::array<Long, SchoolType::total>> num_infected_teach_sch(params.num_diseases, {0});

    amrex::ParmParse::QueryUnusedInputs();

    {
        BL_PROFILE_REGION("Evolution");
        for (int i = 0; i < params.nsteps; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            if ((params.plot_int > 0) && (i % params.plot_int == 0)) {
                if (params.ic_type == ICType::Census) {
                    ExaEpi::IO::writePlotFile(pc, disease_stats, &censusData.unit_mf, &censusData.FIPS_mf, &censusData.comm_mf,
                                              params.num_diseases, params.disease_names, cur_time, i);
                } else {
                    ExaEpi::IO::writePlotFile(pc, disease_stats, nullptr, &urbanPopData.geoid_mf, &urbanPopData.community_mf,
                                              params.num_diseases, params.disease_names, cur_time, i);
                }
            }

            if ((params.aggregated_diag_int > 0) && (i % params.aggregated_diag_int == 0)) {
                if (params.ic_type == ICType::Census) {
                    ExaEpi::IO::writeFIPSData(pc, censusData, params.aggregated_diag_prefix, params.num_diseases,
                                              params.disease_names, i);
                } else {
                    ExaEpi::IO::writeAggregatedData(pc, urbanPopData, params.aggregated_diag_prefix, params.num_diseases,
                                                    params.disease_names, i);
                }
            }

            // Update agents' disease status
            pc.updateStatus(disease_stats);

            for (int d = 0; d < params.num_diseases; d++) {
                auto counts = pc.getTotals(d);

                // Update peak infections
                if (counts[1] > num_infected_peak[d]) {
                    num_infected_peak[d] = counts[1];
                    step_of_peak[d] = i;
                }
                cumulative_deaths[d] = counts[4];
                num_infected[d] = counts[1];
                cum_num_infected[d] += counts[1];

                auto counts_work = pc.getTotalsWorkers(d);
                if (counts_work[1] > num_infected_peak_work[d]) {
                    num_infected_peak_work[d] = counts_work[1];
                }
                cumulative_deaths_work[d] = counts_work[4];
                num_infected_work[d] = counts_work[1];
                cum_num_infected_work[d] += counts_work[1];

                auto counts_teach = pc.getTotalsTeachers(d);
                if (counts_teach[1] > num_infected_peak_teachers[d]) {
                    num_infected_peak_teachers[d] = counts_teach[1];
                }
                cumulative_deaths_teachers[d] = counts_teach[4];
                num_infected_teachers[d] = counts_teach[1];
                cum_num_infected_teachers[d] += counts_teach[1];

                auto counts_nonwork = pc.getTotalsNonWorkers(d);
                if (counts_nonwork[1] > num_infected_peak_nonwork[d]) {
                    num_infected_peak_nonwork[d] = counts_nonwork[1];
                }
                cumulative_deaths_nonwork[d] = counts_nonwork[4];
                num_infected_nonwork[d] = counts_nonwork[1];
                cum_num_infected_nowork[d] += counts_nonwork[1];

                auto counts_student = pc.getTotalsStudent(d);
                if (counts_student[1] > num_infected_peak_student[d]) {
                    num_infected_peak_student[d] = counts_student[1];
                }
                cumulative_deaths_student[d] = counts_student[4];
                num_infected_student[d] = counts_student[1];
                cum_num_infected_student[d] += counts_student[1];

                // Loop over all age groups
                for (int ag = 0; ag < AgeGroups::total; ag++) {
                    auto counts_ag = pc.getTotalsAgeGroup(d, ag);
                    if (counts_ag[1] > num_infected_peak_ag[d][ag]) {
                        num_infected_peak_ag[d][ag] = counts_ag[1];
                    }
                    cumulative_deaths_ag[d][ag] = counts_ag[4];
                    num_infected_ag[d][ag] = counts_ag[1];
                    cum_num_infected_ag[d][ag] += counts_ag[1];

                }
                //Loop over school type
                for (int sch = 0; sch < SchoolType::total; sch++) {
                    auto counts_stud_sch = pc.getTotalsSchoolStudent(d, sch);
                    if (counts_stud_sch[1] > num_infected_peak_stud_school[d][sch]) {
                        num_infected_peak_stud_school[d][sch] = counts_stud_sch[1];
                    }
                    cumulative_deaths_stud_school[d][sch] = counts_stud_sch[4];
                    num_infected_stud_sch[d][sch] = counts_stud_sch[1];
                    cum_num_infected_stud_sch[d][sch] += counts_stud_sch[1];
                }
                // Loop over school type
                for (int sch = 1; sch < SchoolType::total; sch++) {
                    auto count_teach_sch = pc.getTotalsTeachersPerSchool(d,sch);
                    if (count_teach_sch[1] > num_infected_peak_teacher_school[d][sch]) {
                        num_infected_peak_teacher_school[d][sch] = count_teach_sch[1];
                    }
                    cumulative_deaths_teacher_school[d][sch] = count_teach_sch[4];
                    num_infected_teach_sch[d][sch] = count_teach_sch[1];
                    cum_num_infected_teach_sch[d][sch] += count_teach_sch[1];
                }

                Real mmc[4] = {0, 0, 0, 0};
#ifdef AMREX_USE_GPU
                if (Gpu::inLaunchRegion()) {
                    auto const& ma = disease_stats[d]->const_arrays();
                    GpuTuple<Real, Real, Real, Real> mm =
                            ParReduce(TypeList<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum>{},
                                      TypeList<Real, Real, Real, Real>{}, *(disease_stats[d]), IntVect(0, 0),
                                      [=] AMREX_GPU_DEVICE (int box_no, int ii, int jj,
                                                           int kk) noexcept -> GpuTuple<Real, Real, Real, Real> {
                                          return {ma[box_no](ii, jj, kk, 0), ma[box_no](ii, jj, kk, 1), ma[box_no](ii, jj, kk, 2),
                                                  ma[box_no](ii, jj, kk, 3)};
                                      });
                    mmc[0] = amrex::get<0>(mm);
                    mmc[1] = amrex::get<1>(mm);
                    mmc[2] = amrex::get<2>(mm);
                    mmc[3] = amrex::get<3>(mm);
                } else
#endif
                {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (!system::regtest_reduction) reduction(+ : mmc[ : 4])
#endif
                    for (MFIter mfi(*(disease_stats[d])); mfi.isValid(); ++mfi) {
                        Box const& bx = mfi.tilebox();
                        auto const& dfab = disease_stats[d]->const_array(mfi);
                        AMREX_LOOP_3D(bx, ii, jj, kk, {
                            mmc[0] += dfab(ii, jj, kk, 0);
                            mmc[1] += dfab(ii, jj, kk, 1);
                            mmc[2] += dfab(ii, jj, kk, 2);
                            mmc[3] += dfab(ii, jj, kk, 3);
                        });
                    }
                }

                ParallelDescriptor::ReduceRealSum(&mmc[0], 4, ParallelDescriptor::IOProcessorNumber());

                if (ParallelDescriptor::IOProcessor()) {
                    // total number of deaths computed on agents and on mesh should be the same...
                    if (mmc[3] != counts[4]) { amrex::Print() << mmc[3] << " " << counts[4] << "\n"; }
                    AMREX_ALWAYS_ASSERT(mmc[3] == counts[4]);

                    // the total number of infected should equal the sum of
                    //     exposed but not infectious
                    //     infectious and asymptomatic
                    //     infectious and pre-symptomatic
                    //     infectious and symptomatic
                    AMREX_ALWAYS_ASSERT(counts[1] == counts[5] + counts[6] + counts[7] + counts[8]);

                    std::ofstream File;
                    File.open(output_filename[d].c_str(), std::ios::out | std::ios::app);

                    if (!File.good()) { amrex::FileOpenFailed(output_filename[d]); }

                    File << std::setw(5) << i;
                    File << std::setw(12) << counts[0];
                    File << std::setw(12) << counts[1];
                    File << std::setw(12) << counts[2];
                    File << std::setw(12) << counts[4];
                    File << std::setw(15) << mmc[0];
                    File << std::setw(15) << mmc[1];
                    File << std::setw(12) << mmc[2];
                    File << std::setw(12) << counts[5];
                    File << std::setw(15) << counts[6];
                    File << std::setw(15) << counts[7];
                    File << std::setw(15) << counts[8] << "\n";

                    File.flush();

                    File.close();

                    if (!File.good()) { amrex::Abort("problem writing output file"); }
                }
            }

            if (params.shelter_start > 0 && params.shelter_start == i) { pc.shelterStart(); }

            if (params.shelter_start > 0 && params.shelter_start + params.shelter_length == i) { pc.shelterStop(); }

            if ((params.random_travel_int > 0) && (i % params.random_travel_int == 0)) {
                pc.moveRandomTravel(params.random_travel_prob);
            }

            if ((params.air_travel_int > 0) && (i % params.air_travel_int == 0)) {
                pc.moveAirTravel(censusData.unit_mf, air, censusData.demo);
            }

            // Typical day
            pc.morningCommute(mask_behavior);
            pc.interactDay(mask_behavior);
            pc.eveningCommute(mask_behavior);
            pc.interactEvening(mask_behavior);
            pc.interactNight(mask_behavior);

            if ((params.random_travel_int > 0) && (i % params.random_travel_int == 0)) { pc.returnRandomTravel(); }

            if ((params.air_travel_int > 0) && (i % params.air_travel_int == 0)) { pc.returnAirTravel(); }

            // Infect agents based on their interactions
            pc.infectAgents(disease_stats);

            std::chrono::duration<double> elapsed_time = std::chrono::high_resolution_clock::now() - start_time;

            Print() << "[Day " << cur_time << " " << std::fixed << std::setprecision(1) << elapsed_time.count()
                    << "s] infected: ";
            for (int d = 0; d < params.num_diseases; d++) {
                if (d > 0) Print() << ", ";
                int total_here = 0;
                Print() << params.disease_names[d] << " " << num_infected[d];
                std::array<std::string, AgeGroups::total> age_group_names = {
                    "Under 5", "5-17", "18-29", "30-49", "50-64", "Over 65"};
                for (int ag = 0; ag < AgeGroups::total; ag++){
                    total_here += num_infected_ag[d][ag];
                    // Print() <<",  " << age_group_names[ag] << ": " << num_infected_ag[d][ag] ;
                }
                // Print() << "; total_here: " << total_here;
            }
            // the cumulative deaths are not tracked separately for each disease
            Print() << "; deaths: " << cumulative_deaths[0] << "\n";
            // pc.printStudentTeacherCountsInfectedDeaths(0);

            cur_time += 1.0_rt; // time step is one day
        }
    }

    if (params.num_diseases == 1) {
        amrex::Print() << "\n \n";
        amrex::Print() << "Peak number of infected: " << num_infected_peak[0] << "\n";
        amrex::Print() << "Day of peak: " << step_of_peak[0] << "\n";
        amrex::Print() << "Cumulative deaths: " << cumulative_deaths[0] << "\n";

        amrex::Print() << "\n \n";
        pc.printWorkerCounts();
        amrex::Print() << "\n \n";

        /* Print Worker/Teachers/Non Workers -- Cumulative Death and Infections*/
        amrex::Print() << "Cumulative deaths non workers: " << cumulative_deaths_nonwork[0] << "\n";
        amrex::Print() << "Cumulative deaths teachers: " << cumulative_deaths_teachers[0] << "\n";
        amrex::Print() << "Cumulative deaths workers: " << cumulative_deaths_work[0] << "\n";
        amrex::Print() << "Peak infected non workers: " << num_infected_peak_nonwork[0] << "\n";
        amrex::Print() << "Peak infected teachers: " << num_infected_peak_teachers[0] << "\n";
        amrex::Print() << "Peak infected workers: " << num_infected_peak_work[0] << "\n";
        // amrex::Print() << "Cumulative infected non workers: " << cum_num_infected_nowork[0] << "\n";
        // amrex::Print() << "Cumulative infected teachers: " << cum_num_infected_teachers[0] << "\n";
        // amrex::Print() << "Cumulative infected workers: " << cum_num_infected_work[0] << "\n";

        /* Print All Agents -- Per Age Group -- Cumulative Death and Infections*/
        amrex::Print() << "Cumulative deaths per age group:\n";
        std::array<std::string, AgeGroups::total> age_group_names = {
            "Under 5", "5-17", "18-29", "30-49", "50-64", "Over 65"
        };
        int total_ag_death = 0;
        int total_stddd_death = 0;
        for (int i = 0; i < AgeGroups::total; i++) {
            amrex::Print() << "  " << age_group_names[i] << ": " << cumulative_deaths_ag[0][i] << "\n";
            total_ag_death += cumulative_deaths_ag[0][i];
            if (i<2){total_stddd_death+= cumulative_deaths_ag[0][i];}
        }
        amrex::Print() <<"  Total(manual): " << total_ag_death << "  Total: " << cumulative_deaths[0]<< "\n \n";

        // amrex::Print() << "Cumulative infected per age group:\n";
        amrex::Print() << "Peak infected per age group:\n";
        int total_ag_infec = 0;
        int total_stddd_infec = 0;
        for (int i = 0; i < AgeGroups::total; i++) {
            amrex::Print() << "  " << age_group_names[i] << ": " << num_infected_peak_ag[0][i] << "\n";
            total_ag_infec += num_infected_peak_ag[0][i];
            if (i<2){total_stddd_infec+=num_infected_peak_ag[0][i];}
            // amrex::Print() << "  " << age_group_names[i] << ": " << cum_num_infected_ag[0][i] << "\n";
            // total_ag_infec += cum_num_infected_ag[0][i];
            // if (i<2){total_stddd_infec+=cum_num_infected_ag[0][i];}
        }
        amrex::Print() << "  Total(manual): " << total_ag_infec << "  Total: " << num_infected_peak[0]<< "\n \n";
        // amrex::Print() << "  Total(manual): " << total_ag_infec << "  Total: " << cum_num_infected[0]<< "\n \n";


         /* Print All Students -- Per School type -- Cumulative Death and Infections*/
        amrex::Print() << "Cumulative deaths students per school type:\n";
        std::array<std::string, SchoolType::total> school_type_names = {
            "none", "college", "high", "middle", "elem", "daycare"
        };
        // std::array<std::string, SchoolType::total> school_type_names = {
        //     "none", "high_1", "middle_2", "elem_3", "elem_4", "daycare"
        // };
        int total_std_sch_death = 0;
        for (int i = 0; i < SchoolType::total; i++) {
            amrex::Print() << "  " << school_type_names[i] << ": " << cumulative_deaths_stud_school[0][i] << "\n";
            total_std_sch_death += cumulative_deaths_stud_school[0][i];
        }
        amrex::Print() << "  Total(manual): " << total_std_sch_death <<"  Total(from AG): " << total_stddd_death << "  Total function: " << cumulative_deaths_student[0]<< "\n \n";

        amrex::Print() << "Peak infected per school type:\n";
        // amrex::Print() << "Cumulative infected per school type:\n";
        int total_std_sch_infec = 0;
        for (int i = 0; i < SchoolType::total; i++) {
            amrex::Print() << "  " << school_type_names[i] << ": " << num_infected_peak_stud_school[0][i] << "\n";
            total_std_sch_infec += num_infected_peak_stud_school[0][i];
            // amrex::Print() << "  " << school_type_names[i] << ": " << cum_num_infected_stud_sch[0][i] << "\n";
            // total_std_sch_infec += cum_num_infected_stud_sch[0][i];
        }
        amrex::Print() << "  Total(manual): " << total_std_sch_infec <<"  Total(from AG): " << total_stddd_infec << "  Total: " << num_infected_peak_student[0]<< "\n \n";
        // amrex::Print() << "  Total(manual): " << total_std_sch_infec <<"  Total(from AG): " << total_stddd_infec << "  Total: " << cum_num_infected_student[0]<< "\n \n";

        /* Print All Teachers -- Per School Type -- Cumulative Death and Infections */
        amrex::Print() << "Cumulative deaths teachers per school type:\n";
        int total_teac_sch_death = 0;
        for (int i = 1; i < SchoolType::total; i++) {
            amrex::Print() << "  " << school_type_names[i] << ": " << cumulative_deaths_teacher_school[0][i] << "\n";
            total_teac_sch_death += cumulative_deaths_teacher_school[0][i];
        }
        amrex::Print() <<  "  Total(manual): " << total_teac_sch_death << "  Total: " << cumulative_deaths_teachers[0] << "\n \n";

        amrex::Print() << "Peak infected teacher per school type:\n";
        // amrex::Print() << "Cumulative infected teacher per school type:\n";
        int total_teac_sch_infec = 0;
        for (int i = 1; i < SchoolType::total; i++) {
            amrex::Print() << "  " << school_type_names[i] << ": " << num_infected_peak_teacher_school[0][i] << "\n";
            total_teac_sch_infec += num_infected_peak_teacher_school[0][i];
            // amrex::Print() << "  " << school_type_names[i] << ": " << cum_num_infected_teach_sch[0][i] << "\n";
            // total_teac_sch_infec += cum_num_infected_teach_sch[0][i];
        }
        amrex::Print() <<  "  Total(manual): " << total_teac_sch_infec << "  Total: " << num_infected_peak_teachers[0] << "\n \n";

        amrex::Print() << "\n \n";

    } else {
        amrex::Print() << "\n \n";
        for (int d = 0; d < params.num_diseases; d++) {
            amrex::Print() << "Disease " << params.disease_names[d] << ":\n";
            amrex::Print() << "    Peak number of infected: " << num_infected_peak[d] << "\n";
            amrex::Print() << "    Day of peak: " << step_of_peak[d] << "\n";
            amrex::Print() << "    Cumulative deaths: " << cumulative_deaths[d] << "\n";

            // Print cumulative infected per age group
            amrex::Print() << "    Cumulative infected per age group:\n";
            std::array<std::string, AgeGroups::total> age_group_names = {
                "Under 5", "5-17", "18-29", "30-49", "50-64", "Over 65"
            };

            for (int i = 0; i < AgeGroups::total; i++) {
                amrex::Print() << "        " << age_group_names[i] << ": " << cum_num_infected_ag[d][i] << "\n";
            }
        }
        amrex::Print() << "\n \n";
    }


    if (params.plot_int > 0) {
        if (params.ic_type == ICType::Census) {
            ExaEpi::IO::writePlotFile(pc, disease_stats, &censusData.unit_mf, &censusData.FIPS_mf, &censusData.comm_mf,
                                      params.num_diseases, params.disease_names, cur_time, params.nsteps);
        } else {
            ExaEpi::IO::writePlotFile(pc, disease_stats, nullptr, &urbanPopData.geoid_mf, &urbanPopData.community_mf,
                                      params.num_diseases, params.disease_names, cur_time, params.nsteps);
        }
    }

    if ((params.aggregated_diag_int > 0) && (params.nsteps % params.aggregated_diag_int == 0)) {
        if (params.ic_type == ICType::Census) {
            ExaEpi::IO::writeFIPSData(pc, censusData, params.aggregated_diag_prefix, params.num_diseases, params.disease_names,
                                      params.nsteps);
        } else {
            ExaEpi::IO::writeAggregatedData(pc, urbanPopData, params.aggregated_diag_prefix, params.num_diseases,
                                            params.disease_names, params.nsteps);
        }
    }
}
