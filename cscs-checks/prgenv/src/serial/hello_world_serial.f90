program hellow
implicit none
write (*,'(A23,1X,I3,1X,A6,1X,I3,1X,A12,1X,I3,1X,A6,I3)') 'Hello World from thread', 0, &
	'out of', 1, 'from process', 0, 'out of', 1
end
