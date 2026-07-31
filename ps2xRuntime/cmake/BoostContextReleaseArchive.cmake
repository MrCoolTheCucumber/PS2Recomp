# Boost's traditional release archives flatten all public headers into the
# top-level boost/ directory.  The component CMakeLists.txt files remain in
# libs/, but their per-component include/ directories are deliberately
# removed.  Adapt that release layout without modifying the fetched source or
# selecting an unversioned system Boost.
function(ps2x_add_boost_context_from_release_archive boost_source_dir)
    set(_expected_boost_version 109100)
    set(_boost_version_header
        "${boost_source_dir}/boost/version.hpp")
    set(_boost_context_cmake
        "${boost_source_dir}/libs/context/CMakeLists.txt")

    if(NOT EXISTS "${_boost_version_header}" OR
       NOT EXISTS "${_boost_context_cmake}")
        message(FATAL_ERROR
            "Pinned Boost release archive has an unexpected layout")
    endif()

    file(STRINGS "${_boost_version_header}" _boost_version_define
        REGEX "^#define BOOST_VERSION [0-9]+$")
    if(NOT _boost_version_define STREQUAL
       "#define BOOST_VERSION ${_expected_boost_version}")
        message(FATAL_ERROR
            "EE fibers require exact Boost 1.91.0; found "
            "'${_boost_version_define}'")
    endif()

    # Boost.Context's direct dependencies are header-only in this release.
    # Give its unchanged component build the canonical target names while
    # sourcing those headers from the archive's flattened include root.
    foreach(_dependency
            assert
            config
            core
            mp11
            pool
            predef
            smart_ptr)
        if(TARGET "Boost::${_dependency}")
            message(FATAL_ERROR
                "Unexpected pre-existing Boost::${_dependency} target")
        endif()

        set(_target "ps2x_boost_${_dependency}_headers")
        add_library("${_target}" INTERFACE)
        target_include_directories("${_target}" INTERFACE
            "$<BUILD_INTERFACE:${boost_source_dir}>")
        add_library("Boost::${_dependency}" ALIAS "${_target}")
    endforeach()

    set(BOOST_SUPERPROJECT_VERSION 1.91.0)
    set(BOOST_SUPERPROJECT_SOURCE_DIR "${boost_source_dir}")
    set(_saved_build_testing "${BUILD_TESTING}")
    set(BUILD_TESTING OFF)
    add_subdirectory(
        "${boost_source_dir}/libs/context"
        "${CMAKE_CURRENT_BINARY_DIR}/boost-context-release"
        EXCLUDE_FROM_ALL)
    set(BUILD_TESTING "${_saved_build_testing}")

    if(NOT TARGET Boost::context)
        message(FATAL_ERROR
            "Pinned Boost release archive did not provide Boost::context")
    endif()

    # Replace the component-local include path, which is absent by design in
    # a traditional release archive, with its consolidated public header root.
    set_target_properties(boost_context PROPERTIES
        INCLUDE_DIRECTORIES "${boost_source_dir}"
        INTERFACE_INCLUDE_DIRECTORIES
            "$<BUILD_INTERFACE:${boost_source_dir}>")
endfunction()
