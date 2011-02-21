require"imlua"
require"imlua_process"

testdir='D:\\users\\fmiranda\\testes'
basedir='tet_linear_pre'
targetdir=''

base = im.FileImageLoad('base.png')
target = im.FileImageLoad('target.png')
--diff = im.ImageCreate(base:Width(), base:Height(), im.RGB, im.BYTE)

--for imagenum = 1, 64, 1 do
	diff = im.ProcessArithmeticOpNew(base, target, 1)
--end

diff:Save("result.png", "PNG")
