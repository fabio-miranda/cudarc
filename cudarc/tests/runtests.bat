@CALL "%VS90COMNTOOLS%\vsvars32.bat"
cd ../../../prj/cudarc/vc9
lua5.1 ../../../src/cudarc/tests/tests.lua
@pause