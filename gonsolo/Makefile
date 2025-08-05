# TODO:
# Remove workaround in nouveau_bo.h for bindgen 66
MESON = meson
BUILD_DIR_NAME = build
BUILD_DIR = ../$(BUILD_DIR_NAME)
#VULKANINFO = ~/src/Vulkan-Tools/gonsolo/vulkaninfo/vulkaninfo
VULKANINFO = /usr/bin/vulkaninfo
VULKAN_COMPUTE_DIR = ~/work/VulkanHpp-Compute-Sample/gonsolo
VULKAN_COMPUTE = $(VULKAN_COMPUTE_DIR)/VulkanCompute
#APPLICATION = $(VULKANINFO)
APPLICATION = $(VULKAN_COMPUTE)
MACHINE = $(shell uname -m)
ICD_D = ~/bin/mesa/share/vulkan/icd.d
NOUVEAU_ICD = $(ICD_D)/nouveau_icd.$(MACHINE).json
BORG_ICD = $(ICD_D)/borg_icd.$(MACHINE).json
ifeq (,$(wildcard /sys/devices/platform/soc/4000.borg-device))
	ICD = $(NOUVEAU_ICD)
else
	ICD = $(BORG_ICD)
endif

#icd:
#	@echo $(ICD)

DRIVERS = nouveau,borg-experimental
#VERBOSE = --verbose
SKIP_DRM = 1

all: compile
t: test
test: test_nouveau
tn: test_nouveau
test_nouveau:
	cd $(VULKAN_COMPUTE_DIR); MESA_SHADER_CACHE_DISABLE=1 NAK_DEBUG=print RUST_BACKTRACE=1 BORG_SKIP_DRM=$(SKIP_DRM) VK_ICD_FILENAMES=$(NOUVEAU_ICD) $(APPLICATION); cd -
tb: test_borg
test_borg:
	cd $(VULKAN_COMPUTE_DIR); MESA_SHADER_CACHE_DISABLE=1 NAK_DEBUG=print RUST_BACKTRACE=1 BORG_SKIP_DRM=$(SKIP_DRM) VK_ICD_FILENAMES=$(BORG_ICD) $(APPLICATION); cd -
gdb: #install
	cd $(VULKAN_COMPUTE_DIR); VK_ICD_FILENAMES=$(ICD) gdb $(APPLICATION); cd -
rust-gdb: #install
	cd $(VULKAN_COMPUTE_DIR); VK_ICD_FILENAMES=$(ICD) rust-gdb $(APPLICATION); cd -
lldb: install
	cd $(VULKAN_COMPUTE_DIR); VK_ICD_FILENAMES=$(ICD) lldb -o run $(APPLICATION); cd -
i: install
install: #compile_and_tidy
	cd $(BUILD_DIR); $(MESON) install; cd -
ct: compile_and_tidy
compile_and_tidy: compile tidy_compile_commands
compile:
	cd $(BUILD_DIR); mold -run $(MESON) compile $(VERBOSE); cd -
link_compile_commands:
	cd ..; ln -s $(BUILD_DIR_NAME)/compile_commands.json .; cd -
setup:
	cd ..; $(MESON) setup $(BUILD_DIR_NAME) -Dbackend=ninja -Dgallium-drivers= \
		-Dvulkan-drivers=$(DRIVERS) -Dprefix=~/bin/mesa \
		-Dbuild-tests=true -Dbuildtype=debug -Db_sanitize=address; cd -
clean:
	cd ..; rm -rf $(BUILD_DIR_NAME); cd -

tidy_compile_commands:
	cd $(BUILD_DIR); sed -i 's/-mtls-dialect=gnu2 //g' compile_commands.json; cd -
	cd $(BUILD_DIR); sed -i 's/-flifetime-dse=1 //g' compile_commands.json; cd -

run_nir_tests:
	meson test -C ../build nir_tests
.PHONY: all clean compile compile_and_tidy ct install link_compile_commands \
	run_nir_tests setup t test
