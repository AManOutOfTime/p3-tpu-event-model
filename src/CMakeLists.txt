# Single library that holds the whole simulator. To add a new source file,
# append it to the SIM_CORE_SOURCES list below -- nothing else to touch.
set(SIM_CORE_SOURCES
    core/event_engine.cpp
    core/logger.cpp
    config/arch_config.cpp
    schedule/schedule.cpp
    schedule/op_registry.cpp
    schedule/scheduler.cpp
    units/printing_unit.cpp
    units/delay_unit.cpp
)

add_library(sim_core STATIC ${SIM_CORE_SOURCES})

target_include_directories(sim_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sim_core
    PUBLIC yaml-cpp::yaml-cpp
)
