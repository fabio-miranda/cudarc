srcdir='..\\..\\..\\src\\cudarc'

--Home
--model='J:\\vis\\models\\nasa\\bluntfin\\bluntfin.grid'
--exepath='J:\\vis\\prj\\cudarc\\vc9\\tcudarc\_Release\\tcudarc.exe'
--args=model..' -outputpath J:\\testes\\ -shaderpath J:\\vis\\src\\cudarc\\glsl\\ -zetapsigammapath J:\\vis\\src\\cudarc\\zetapsigamma\\ -benchmark'

--Tec
model='D:\\users\\fmiranda\\vis\\models\\nasa\\bluntfin\\bluntfin.grid'
exepath='D:\\users\\fmiranda\\vis\\prj\\cudarc\\vc9\\tcudarc\_Release\\tcudarc.exe'
args=model..' -outputpath D:\\users\\fmiranda\\testes\\ -shaderpath D:\\users\\fmiranda\\vis\\src\\cudarc\\glsl\\ -zetapsigammapath D:\\users\\fmiranda\\vis\\src\\cudarc\\\zetapsigamma\\ -benchmark'

--We have to force rebuild because of CUDA custom build rules
clean='Devenv ..\\..\\..\\prj\\cudarc\\vc9\\cudarc.sln /rebuild Release /project libcudarc.vcproj /projectconfig Release'
build1='Devenv /build Release ..\\..\\..\\prj\\cudarc\\vc9\\cudarc.sln /project tcudarc.vcproj /projectconfig Release'
build2='Devenv /build Release ..\\..\\..\\prj\\cudarc\\vc9\\cudarc.sln /project tcudarc.vcproj /projectconfig Release'

function write(str)
defaults = '#define CUDARC_FEM\n#define CUDARC_RES\n#define CUDARC_VERBOSE\n#define CUDARC_TIME\n'
file = io.open(srcdir..'\\defines.h', "w")
file:write(defaults)
file:write(str)
file:close(srcdir..'defines.h')
end

--os.execute('@ECHO Runing exe: '..exepath..' '..args)
--os.execute(exepath..' '..args)

-- Test 1
os.execute('@ECHO Test 1: Tet. and cte. integration')
write('#define CUDARC_CONST\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 1')

-- Test 2
os.execute('@ECHO Test 2: Tet. and cte. integration')
write('')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 2')

-- Test 3
os.execute('@ECHO Test 3: Tet. and cte. integration')
write('#define CUDARC_CALCULATE_ZETAPSI\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 3')

-- Test 4
os.execute('@ECHO Test 4: Tet. and cte. integration')
write('#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 5\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 4')

-- Test 5
os.execute('@ECHO Test 5: Tet. and cte. integration')
write('#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 10\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 5')

-- Test 6
os.execute('@ECHO Test 6: Tet. and cte. integration')
write('#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 20\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 6')

-- Test 7
os.execute('@ECHO Test 7: Tet. and cte. integration')
write('#define CUDARC_HEX\n#define CUDARC_CONST\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 7')

-- Test 8
os.execute('@ECHO Test 8: Tet. and cte. integration')
write('#define CUDARC_HEX\n#define CUDARC_CALCULATE_ZETAPSI\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 8')

-- Test 9
os.execute('@ECHO Test 9: Tet. and cte. integration')
write('#define CUDARC_HEX\n#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 5\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 9')

-- Test 10
os.execute('@ECHO Test 10: Tet. and cte. integration')
write('#define CUDARC_HEX\n#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 10\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 10')

-- Test 11
os.execute('@ECHO Test 11: Tet. and cte. integration')
write('#define CUDARC_HEX\n#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 20\n')
os.execute(clean)
os.execute(build1)
os.execute(build2)
os.execute('@ECHO Runing exe: '..exepath..' '..args)
os.execute(exepath..' '..args)
os.execute('@ECHO End test 11')
