include(CheckIPOSupported)

check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

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
    else()
        message(WARNING "Interprocedural optimization not supported: ${ipo_error}")
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
