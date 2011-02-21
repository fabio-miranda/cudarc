----
-- Fabio Markus Miranda
-- fmiranda@tecgraf.puc-rio.br
-- fabiom@gmail.com
-- Dec 2010
-- 
-- runtests.lua: compile and run t.cudarc.exe
----

function write(srcdir, str)
	local defaults = '#define CUDARC_FEM\n#define CUDARC_RES\n#define CUDARC_VERBOSE\n#define CUDARC_TIME\n#define CUDARC_WHITE\n'
	local file = io.open(srcdir..'\\defines.h', "w")
	file:write(defaults)
	file:write(str)
	file:close(srcdir..'defines.h')
end

function runtests(srcdir, model, exepath, args, defineslist)

	--We have to force rebuild because of CUDA custom build rules
	local clean='Devenv ..\\..\\..\\prj\\cudarc\\vc9\\cudarc.sln /rebuild Release /project libcudarc.vcproj /projectconfig Release'
	local build1='Devenv /build Release ..\\..\\..\\prj\\cudarc\\vc9\\cudarc.sln /project tcudarc.vcproj /projectconfig Release'
	local build2='Devenv /build Release ..\\..\\..\\prj\\cudarc\\vc9\\cudarc.sln /project tcudarc.vcproj /projectconfig Release'

	--os.execute('@ECHO Runing exe: '..exepath..' '..args)
	--os.execute(exepath..' '..args)

	for index, defines in ipairs(defineslist) do
		
		write(srcdir, defines)
		os.execute(clean)
		os.execute(build1)
		os.execute(build2)
		os.execute('@ECHO Runing exe: '..exepath..' '..args)
		os.execute(exepath..' '..args)
	
	end

end
