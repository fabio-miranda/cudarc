----
-- Fabio Markus Miranda
-- fmiranda@tecgraf.puc-rio.br
-- fabiom@gmail.com
-- Dec 2010
-- 
-- generatelogs.lua: call the functions responsible to generate the logs
----

require "logdiff"
require "logtime"
require "runtests"


srcdir='..\\..\\..\\src\\cudarc\\'
modelbasepath='D:\\users\\fmiranda\\vis\\models\\sci_data\\'
testbasedir='D:\\users\\fmiranda\\testes\\'
exepath='D:\\users\\fmiranda\\vis\\prj\\cudarc\\vc9\\tcudarc\_Release\\tcudarc.exe'

models={'hex8_1x1x1_1.node',
		'hex8_1x1x1_2.node',
		'hex8_1x1x1_3.node',
		'hex8_1x1x1_4.node',
		'hex8_1x1x1_2op.node'
		}
		
tfs={'simple_const.tf',
	'simple_transp_const.tf',
	'simple_linear.tf',
	'simple_transp_linear.tf'
	}
	
defines={
	'#define CUDARC_CONST\n',
	'',
	'#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 5\n',
	'#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 20\n',
	'#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 100\n',
	'#define CUDARC_HEX\n#define CUDARC_CONST\n',
	'#define CUDARC_HEX\n#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 5\n',
	'#define CUDARC_HEX\n#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 20\n',
	'#define CUDARC_HEX\n#define CUDARC_INTEGRATE_FIXEDSTEPS\n#define CUDARC_NUM_STEPS 100\n'
	}

for index, modelname in ipairs(models) do
	
	for index, tfname in ipairs(tfs) do
		outputpath = testbasedir..modelname..'\\'..tfname..'\\'
		modelpath = modelbasepath..modelname
		args = modelpath..' -shaderpath '..srcdir..'glsl\\ -zetapsigammapath '..srcdir..'zetapsigamma\\ -benchmark  -outputpath '..outputpath..' -tfpath '..srcdir..'tf\\'..tfname
		os.execute('mkdir '..outputpath)
		runtests(srcdir, modelpath, exepath, args, defines)
	end
end

