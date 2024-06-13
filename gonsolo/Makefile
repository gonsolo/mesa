#MESON = ~/src/meson/meson.py
MESON = meson
BUILD_DIR_NAME = build
BUILD_DIR = ../$(BUILD_DIR_NAME)
#VULKANINFO = ~/src/Vulkan-Tools/gonsolo/vulkaninfo/vulkaninfo
VULKANINFO = /usr/bin/vulkaninfo
VULKAN_COMPUTE_DIR = ~/src/VulkanHpp-Compute-Sample/gonsolo
VULKAN_COMPUTE = ~/src/VulkanHpp-Compute-Sample/gonsolo/VulkanCompute
#APPLICATION = $(VULKANINFO)
APPLICATION = $(VULKAN_COMPUTE)
MACHINE = $(shell uname -m)
ICD = ~/bin/mesa/share/vulkan/icd.d/borg_icd.$(MACHINE).json
DRIVERS = nouveau,borg-experimental
#DRIVERS = borg-experimental

test: #install
	cd $(VULKAN_COMPUTE_DIR); VK_ICD_FILENAMES=$(ICD) $(APPLICATION); cd -
gdb: #install
	cd $(VULKAN_COMPUTE_DIR); VK_ICD_FILENAMES=$(ICD) gdb $(APPLICATION); cd -
lldb: install
	cd $(VULKAN_COMPUTE_DIR); VK_ICD_FILENAMES=$(ICD) lldb -o run $(APPLICATION); cd -
install: compile_and_tidy
	cd $(BUILD_DIR); $(MESON) install; cd -
compile_and_tidy: compile tidy_compile_commands
compile:
	cd $(BUILD_DIR); $(MESON) compile; cd -
link_compile_commands:
	cd ..; ln -s $(BUILD_DIR_NAME)/compile_commands.json .; cd -
setup:
	cd ..; $(MESON) setup $(BUILD_DIR_NAME) -Dbackend=ninja -Dgallium-drivers= -Dvulkan-drivers=$(DRIVERS) -Dprefix=~/bin/mesa; cd -
c: clean
clean:
	cd ..; rm -rf $(BUILD_DIR_NAME); cd -

tidy_compile_commands:
	cd $(BUILD_DIR); sed -i 's/-mtls-dialect=gnu2 //g' compile_commands.json; cd -
	cd $(BUILD_DIR); sed -i 's/-flifetime-dse=1 //g' compile_commands.json; cd -

.PHONY: c clean compile install setup test

bla: bla.c
	gcc -I/usr/include/libdrm -ldrm -o bla bla.c
