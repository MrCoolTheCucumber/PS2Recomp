include(CheckIPOSupported)

check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

if(PS2X_ENABLE_GCC_INCREMENTAL_LTO)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR
            "PS2X_ENABLE_GCC_INCREMENTAL_LTO requires GCC")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
        message(FATAL_ERROR
            "PS2X_ENABLE_GCC_INCREMENTAL_LTO requires GCC 16 or newer")
    endif()

    get_filename_component(PS2X_GCC_INCREMENTAL_LTO_CACHE_DIR
        "${PS2X_GCC_INCREMENTAL_LTO_CACHE_DIR}"
        ABSOLUTE
        BASE_DIR "${CMAKE_BINARY_DIR}")
    file(MAKE_DIRECTORY "${PS2X_GCC_INCREMENTAL_LTO_CACHE_DIR}")
endif()

function(EnableFastReleaseMode TargetName)
    message("> Enabling optimization for: ${TargetName}")
    if(MSVC)
        target_compile_options(${TargetName} PRIVATE
            $<$<CONFIG:Release>:
                /O2 # speed
                /Ob2 # inline aggressively
                /Oi # intrinsics
                /GL # whole program opt
                /Gy # function-level linking
                /Gw # global data in COMDAT
                /GF # string pooling
                /Zc:inline # remove unreferenced inline
                /fp:fast # fast math (graphics friendly)
                /DNDEBUG
                /arch:AVX2 # Advanced Vector Extensions 2
                /GS- # Disable Buffer Security Check (faster)
                /Qspectre- # Disable Spectre mitigations (faster)
            >
        )

        if(TARGET ${TargetName})
            target_link_options(${TargetName} PRIVATE
                $<$<CONFIG:Release>:
                    /LTCG # link-time code generation
                    /OPT:REF # remove unreferenced
                    /OPT:ICF # fold identical COMDATs
                >
            )
        endif()
    endif()

    if(IPO_SUPPORTED)
        set_property(TARGET ${TargetName} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

        if(PS2X_ENABLE_GCC_INCREMENTAL_LTO)
            get_target_property(PS2X_RELEASE_TARGET_TYPE
                ${TargetName} TYPE)
            if(NOT PS2X_RELEASE_TARGET_TYPE STREQUAL "STATIC_LIBRARY" AND
               NOT PS2X_RELEASE_TARGET_TYPE STREQUAL "OBJECT_LIBRARY" AND
               NOT PS2X_RELEASE_TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
                string(MAKE_C_IDENTIFIER "${TargetName}"
                    PS2X_INCREMENTAL_LTO_TARGET_NAME)
                set(PS2X_TARGET_INCREMENTAL_LTO_CACHE_DIR
                    "${PS2X_GCC_INCREMENTAL_LTO_CACHE_DIR}/${PS2X_INCREMENTAL_LTO_TARGET_NAME}")
                file(MAKE_DIRECTORY
                    "${PS2X_TARGET_INCREMENTAL_LTO_CACHE_DIR}")
                target_link_options(${TargetName} PRIVATE
                    "$<$<CONFIG:Release>:-flto-incremental=${PS2X_TARGET_INCREMENTAL_LTO_CACHE_DIR}>")
                message(STATUS
                    "GCC incremental LTO cache for ${TargetName}: "
                    "${PS2X_TARGET_INCREMENTAL_LTO_CACHE_DIR}")
            endif()
        endif()
    else()
        message(WARNING "Interprocedural optimization not supported: ${IPO_ERROR}")
    endif()
endfunction()

# Boost.Context documents /EHs as part of its Windows fcontext contract.
# Suspended frames must also remain outside whole-program optimization.
function(ConfigureEeFcontextSuspendedFrames TargetName)
    if(MSVC AND
       PS2X_ENABLE_EE_CPP_FIBER_BACKEND AND
       CMAKE_SIZEOF_VOID_P EQUAL 8 AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        target_compile_options(${TargetName} PRIVATE
            /EHs
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:/GL->
        )
        target_link_options(${TargetName} PRIVATE
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:/LTCG:OFF>
        )
        set_target_properties(${TargetName} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION FALSE
            INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
            INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO FALSE
        )
    endif()
endfunction()
