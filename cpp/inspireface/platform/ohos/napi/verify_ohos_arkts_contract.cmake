if(NOT DEFINED ISF_NAPI_SOURCE OR NOT DEFINED ISF_NAPI_DECLARATION OR NOT DEFINED ISF_ARKTS_WRAPPER)
    message(FATAL_ERROR "Node-API source, declaration, and ArkTS wrapper paths are required")
endif()

file(READ "${ISF_NAPI_SOURCE}" napi_source)
file(READ "${ISF_NAPI_DECLARATION}" napi_declaration)
file(READ "${ISF_ARKTS_WRAPPER}" arkts_wrapper)

set(expected_exports
        launch
        reload
        terminate
        isLaunched
        getVersion
        getCapiLevel
        createSession
        releaseSession
        configureSession
        clearTracking
        createImageStream
        releaseImageStream
        track
        getDenseLandmarks
        getFiveKeyPoints
        extractFeature
        getFeatureLength
        compareFeatures
        getRecommendedThreshold
        similarityToPercentage
        setLogLevel
        disableLog)

foreach(export_name IN LISTS expected_exports)
    string(FIND "${napi_source}" "{\"${export_name}\"," native_offset)
    if(native_offset EQUAL -1)
        message(FATAL_ERROR "Native Node-API export is missing: ${export_name}")
    endif()

    string(FIND "${napi_declaration}" "${export_name}(" declaration_offset)
    if(declaration_offset EQUAL -1)
        message(FATAL_ERROR "Node-API type declaration is missing: ${export_name}")
    endif()

    string(FIND "${arkts_wrapper}" "native.${export_name}(" wrapper_offset)
    if(wrapper_offset EQUAL -1)
        message(FATAL_ERROR "ArkTS wrapper does not route export: ${export_name}")
    endif()
endforeach()

message(STATUS "Verified ${expected_exports} Node-API/ArkTS contract")
