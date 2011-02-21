----
-- Fabio Markus Miranda
-- fmiranda@tecgraf.puc-rio.br
-- fabiom@gmail.com
-- Dec 2010
-- 
-- logtime.lua: calculate average time
----

function logtime(testdir)

	print('Starting log time')
	local file = io.open(testdir..'avgtime.txt', 'w')
	for index, dir in ipairs(targetdir) do
		local times = io.open(testdir..dir..'\\benchmark.txt')
		local count = 0
		local sum = 0
		local line = times:read()
		while(line:sub(1,1) == '#') do line = times:read() end
		while(line) do
			sum = sum + string.format(line)
			count = count + 1
			line = times:read()
		end
		
		print(dir..' '..sum/count)
		file:write(dir..' '..sum/count ..'\n')
	end
	file:close()

end


function plottime(testdir)
	local gnu = io.open('plottime.gnu')
	t = gnu:read("*all")
	t = string.gsub(t, "$TESTDIR", testdir)
	
	local aux = io.open('plottime.aux', 'w')
	aux:write(t)
	aux:close()
	print('gnuplot plottime.aux')
	os.execute('gnuplot plottime.aux')
	os.execute('del plottime.aux')
	os.execute('mv plottime.png '..testdir)
	os.execute('mv plottime.eps '..testdir)
end