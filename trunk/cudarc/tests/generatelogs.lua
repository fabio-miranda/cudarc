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

srcdir='..\\..\\..\\src\\cudarc'

--Tec
--testdir='D:\\users\\fmiranda\\testes\\'
--model='D:\\users\\fmiranda\\vis\\models\\nasa\\bluntfin\\bluntfin.grid'
--exepath='D:\\users\\fmiranda\\vis\\prj\\cudarc\\vc9\\tcudarc\_Release\\tcudarc.exe'
--args=model..' -outputpath D:\\users\\fmiranda\\testes\\ -shaderpath D:\\users\\fmiranda\\vis\\src\\cudarc\\glsl\\ -zetapsigammapath D:\\users\\fmiranda\\vis\\src\\cudarc\\\zetapsigamma\\ -benchmark'

--Home
--testdir='J:\\\\testes\_8800gtx\\\\'
testdir='J:\\\\testes\\\\'
model='J:\\vis\\models\\nasa\\bluntfin\\bluntfin.grid'
exepath='J:\\vis\\prj\\cudarc\\vc9\\tcudarc\_Release\\tcudarc.exe'
args=model..' -outputpath '..testdir..' -shaderpath J:\\vis\\src\\cudarc\\glsl\\ -zetapsigammapath J:\\vis\\src\\cudarc\\zetapsigamma\\ -benchmark'


basedir='tet_pre'
targetdir={'hex_const', 'hex_gauss', 'hex_fixed5', 'hex_fixed10', 'hex_fixed20', 'tet_const', 'tet_gauss', 'tet_pre', 'tet_fixed5', 'tet_fixed10', 'tet_fixed20'}

runtests(srcdir, model, exepath, args)
logtime(testdir)
plottime(testdir)
logdiff(testdir, basedir, targetdir)

