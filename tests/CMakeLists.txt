# Single test binary built from doctest_main.cpp + every test_*.cpp here.
# To add a test file: just create tests/test_<thing>.cpp and add it below.
set(SIM_TEST_SOURCES
    doctest_main.cpp
    test_event_engine.cpp
    test_config.cpp
    test_schedule.cpp
    test_dummy_units.cpp
    test_fa2_schedule.cpp
    test_systolic.cpp
    test_systolic_compute.cpp
    test_hardware_units.cpp
    test_tiler.cpp
)

add_executable(unit_tests ${SIM_TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE sim_core doctest::doctest)

# Bake the project root into the test binary so tests can load real YAML files
# via Schedule::from_yaml_file() regardless of where the binary is run from.
target_compile_definitions(unit_tests PRIVATE
    SIM_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

add_test(NAME unit_tests COMMAND unit_tests
         WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
