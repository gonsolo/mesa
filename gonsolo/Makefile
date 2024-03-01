MESON = ~/src/meson/meson.py
BUILD_DIR_NAME = gonsolo-nouveau-borg
BUILD_DIR = ../$(BUILD_DIR_NAME)
VULKANINFO = ~/src/Vulkan-Tools/gonsolo/vulkaninfo/vulkaninfo
COMPUTE_SAMPLE = ~/src/VulkanHpp-Compute-Sample/gonsolo/VulkanCompute
#APPLICATION = $(VULKANINFO)
APPLICATION = $(COMPUTE_SAMPLE)

test: install
	VK_ICD_FILENAMES=~/bin/mesa/share/vulkan/icd.d/borg_icd.x86_64.json $(APPLICATION)
gdb: install
	VK_ICD_FILENAMES=~/bin/mesa/share/vulkan/icd.d/borg_icd.x86_64.json gdb $(APPLICATION)
lldb: install
	VK_ICD_FILENAMES=~/bin/mesa/share/vulkan/icd.d/borg_icd.x86_64.json lldb $(APPLICATION)
install: compile_and_tidy
	cd $(BUILD_DIR); $(MESON) install; cd -
compile_and_tidy: compile tidy_compile_commands
compile:
	cd $(BUILD_DIR); $(MESON) compile; cd -
	cd ..; ln -s $(BUILD_DIR_NAME)/compile_commands.json .; cd -
setup:
	cd ..; $(MESON) setup $(BUILD_DIR_NAME) -Dbackend=ninja -Dgallium-drivers= -Dvulkan-drivers=nouveau-experimental,borg-experimental -Dprefix=~/bin/mesa; cd -
c: clean
clean:
	cd ..; rm -rf gonsolo-nouveau-borg; cd -

tidy_compile_commands:
	cd $(BUILD_DIR); sed -i 's/-mtls-dialect=gnu2 //g' compile_commands.json; cd -
	cd $(BUILD_DIR); sed -i 's/-flifetime-dse=1 //g' compile_commands.json; cd -

.PHONY: c clean compile install setup test
