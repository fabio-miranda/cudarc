----
-- Fabio Markus Miranda
-- fmiranda@tecgraf.puc-rio.br
-- fabiom@gmail.com
-- Dec 2010
-- 
-- logdiff.lua: calculate rms error
----

require "imlua"
require"imlua_process"


function logdiff(testdir, basedir, targetdir)

	print('Starting diff images')
	local file = io.open(testdir..'rmserror.txt', 'w')
	for index, dir in ipairs(targetdir) do
		sum = 0
		for imagenum = 1, 64, 1 do
			target = im.FileImageLoad(testdir..basedir..'\\'..imagenum..'.png')
			base = im.FileImageLoad(testdir..dir..'\\'..imagenum..'.png')
			diff = im.CalcRMSError(target, base)
			sum = sum + diff
		end
		--print(dir..' '..sum/64)
		file:write(dir..' '..sum/64 ..'\n')
	end
	file:close()

end
