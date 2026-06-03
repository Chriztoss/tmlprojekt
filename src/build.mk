## Standard behavior must be included here
INCLUDE_DIRS += $(SOURCE_PATH)/$(USRSRC)
CPPSRC += $(call target_files,$(USRSRC_SLASH),*.cpp)
CSRC += $(call target_files,$(USRSRC_SLASH),*.c)

APPSOURCES=$(call target_files,$(USRSRC_SLASH),*.cpp)
APPSOURCES+=$(call target_files,$(USRSRC_SLASH),*.c)

## emlearn include path
INCLUDE_DIRS += C:/Users/chris/Desktop/TML/akkorddata/tmlprojekt/.venv/Lib/site-packages/emlearn