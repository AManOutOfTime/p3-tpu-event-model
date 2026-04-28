# Single test binary built from doctest_main.cpp + every test_*.cpp here.
# To add a test file: just create tests/test_<thing>.cpp and add it below.
set(SIM_TEST_SOURCES
    doctest_main.cpp
    test_event_engine.cpp
    test_config.cpp
    test_schedule.cpp
    test_dummy_units.cpp
)

add_executable(unit_tests ${SIM_TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE sim_core doctest::doctest)

add_test(NAME unit_tests COMMAND unit_tests)
