library ieee;

use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity learningTop is
	generic (
		d           : integer := 1000;
		num_classes : integer := 10
	);
	port (
		clk                  : in std_logic;
		rst                  : in std_logic;
		run                  : in std_logic;
		correct_label        : in integer range 0 to num_classes - 1; -- ground truth label
		predicted_label      : in integer range 0 to num_classes - 1; -- predicted label
		similarity_correct   : in integer range 0 to d;               -- hamming distance to correct class vector
		similarity_incorrect : in integer range 0 to d;               -- hamming distance to predicted class vector
		qhv                  : in std_logic_vector(0 to d - 1);       -- wrongly predicted query vector
		binary_correct       : out std_logic_vector(0 to d - 1);      -- binarized updated correct class weights
		binary_predicted     : out std_logic_vector(0 to d - 1);      -- binarized updated predicted class weights
		done                 : out std_logic
	);
end entity;

architecture behavioral of learningTop is
	function clog2(n : integer) return integer is
		variable i : integer := 0;
		variable v : integer := n - 1;
	begin
		while v > 0 loop
			v := v / 2;
			i := i + 1;
		end loop;
		return i;
	end function;

	constant addr_w : integer := clog2(d * num_classes);

	component popCount is
		generic (lenPop : integer := 8); -- bit width out popCounters
		port (
			clk  : in std_logic;
			rst  : in std_logic;
			en   : in std_logic;
			dout : out std_logic_vector(lenPop - 1 downto 0)
		);
	end component;

	-- This or BRAM in Block Design
	type shadow_vector is array (0 to d - 1) of integer;

	signal update_correct   : shadow_vector := (others => 0);
	signal update_predicted : shadow_vector := (others => 0);
	attribute ram_style : string;
	attribute ram_style of update_correct   : signal is "distributed";
	attribute ram_style of update_predicted : signal is "distributed";
	signal update_completed : std_logic := '0';
	signal current_idx      : integer := 0;
	signal ID               : std_logic_vector(19 downto 0);
	signal rst_counter      : std_logic;
	signal id_at_last       : std_logic;
	signal counter_en       : std_logic;

	component blk_mem_gen_LEARN is
		port (
			clka  : in std_logic;
			wea   : in std_logic;
			addra : in std_logic_vector(addr_w - 1 downto 0);
			dina  : in std_logic_vector(31 downto 0);
			douta : out std_logic_vector(31 downto 0);
			clkb  : in std_logic;
			web   : in std_logic;
			addrb : in std_logic_vector(addr_w - 1 downto 0);
			dinb  : in std_logic_vector(31 downto 0);
			doutb : out std_logic_vector(31 downto 0)
		);
	end component;

	signal learn_addra : std_logic_vector(addr_w - 1 downto 0);
	signal learn_addrb : std_logic_vector(addr_w - 1 downto 0);
	signal read_addra  : std_logic_vector(addr_w - 1 downto 0);
	signal read_addrb  : std_logic_vector(addr_w - 1 downto 0);
	signal write_addra : std_logic_vector(addr_w - 1 downto 0) := (others => '0');
	signal write_addrb : std_logic_vector(addr_w - 1 downto 0) := (others => '0');
	signal learn_dina  : std_logic_vector(31 downto 0) := (others => '0');
	signal learn_dinb  : std_logic_vector(31 downto 0) := (others => '0');
	signal learn_douta : std_logic_vector(31 downto 0);
	signal learn_doutb : std_logic_vector(31 downto 0);
	signal learn_wea   : std_logic := '0';
	signal learn_web   : std_logic := '0';

	signal pipe_valid_s1, pipe_valid_s2, pipe_valid_s3 : std_logic := '0';
	signal pipe_idx_s1, pipe_idx_s2, pipe_idx_s3       : integer range 0 to d - 1;

	signal delta_correct_s1   : signed(31 downto 0);
	signal delta_predicted_s1 : signed(31 downto 0);
	signal value_correct_s1   : signed(31 downto 0);
	signal value_predicted_s1 : signed(31 downto 0);

	signal product_correct_s2   : signed(31 downto 0);
	signal product_predicted_s2 : signed(31 downto 0);
	signal value_correct_s2     : signed(31 downto 0);
	signal value_predicted_s2   : signed(31 downto 0);

	signal wb_idx    : integer range 0 to d - 1 := 0;
	signal wb_active : std_logic                 := '0';
	signal wb_actived : std_logic                 := '0'; --delay wb_active by one cycle, needed to ensure last index is accessed in writeback


	signal id_d1      : integer range 0 to d - 1 := 0;
	signal bram_valid : std_logic                 := '0';
begin
	process (clk)
		variable idx              : integer range 0 to d - 1;
		variable delta_c, delta_p : signed(31 downto 0);
		variable prod_c, prod_p   : signed(47 downto 0); -- wider for multiply
	begin
		if rising_edge(clk) then
			if rst = '1' then
				rst_counter      <= '1';
				update_completed <= '0';
				done             <= '0';
				bram_valid       <= '0';
				pipe_valid_s1    <= '0';
				pipe_valid_s2    <= '0';
				pipe_valid_s3    <= '0';

			elsif run = '1' and update_completed = '0' then
				rst_counter   <= '0';
				id_d1         <= to_integer(unsigned(ID));
				bram_valid    <= '1';
				pipe_valid_s1 <= bram_valid;
				pipe_idx_s1   <= id_d1;

				delta_c := to_signed(similarity_correct, 32);
				if qhv(id_d1) = '0' then
					delta_c := -delta_c;
				end if;
				delta_correct_s1 <= delta_c;
				value_correct_s1 <= signed(learn_douta);

				delta_p := to_signed(similarity_incorrect, 32);
				if qhv(id_d1) = '1' then
					delta_p := -delta_p;
				end if;
				delta_predicted_s1 <= delta_p;
				value_predicted_s1 <= signed(learn_doutb);

				pipe_valid_s2 <= pipe_valid_s1;
				pipe_idx_s2   <= pipe_idx_s1;

				-- shift left instead of multiply
				product_correct_s2   <= shift_left(delta_correct_s1, 6);
				product_predicted_s2 <= shift_left(delta_predicted_s1, 6);
				value_correct_s2     <= value_correct_s1;
				value_predicted_s2   <= value_predicted_s1;

				pipe_valid_s3 <= pipe_valid_s2;
				pipe_idx_s3   <= pipe_idx_s2;
				if pipe_valid_s2 = '1' then
					-- divide by 1024
					update_correct(pipe_idx_s2) <= to_integer(
						product_correct_s2 / 1024 + value_correct_s2
					);
					update_predicted(pipe_idx_s2) <= to_integer(
						product_predicted_s2 / 1024 + value_predicted_s2
					);
				end if;

				if pipe_valid_s3 = '1' and pipe_idx_s3 = d - 1 then
					rst_counter      <= '1';
					update_completed <= '1';
					done             <= '0';
				end if;

			elsif update_completed = '1' and wb_active = '0' then
				wb_active <= '1';
				wb_idx    <= 0;

			elsif wb_active = '1' then
				learn_wea <= '1';
				learn_web <= '1';
				write_addra <= std_logic_vector(
					to_unsigned(correct_label * d + wb_idx, addr_w)
				);
				write_addrb <= std_logic_vector(
					to_unsigned(predicted_label * d + wb_idx, addr_w)
				);
				learn_dina <= std_logic_vector(to_signed(update_correct(wb_idx), 32));
				learn_dinb <= std_logic_vector(to_signed(update_predicted(wb_idx), 32));

				if update_correct(wb_idx) >= 0 then
					binary_correct(wb_idx) <= '1';
				else
					binary_correct(wb_idx) <= '0';
				end if;

				if update_predicted(wb_idx) >= 0 then
					binary_predicted(wb_idx) <= '1';
				else
					binary_predicted(wb_idx) <= '0';
				end if;
				
                if wb_actived = '1' then
				    wb_actived <= '0';
				    wb_active <= '0';
				    learn_wea <= '0';
					learn_web <= '0';
					done             <= '1';
					update_completed <= '0';
					pipe_valid_s1    <= '0';
					pipe_valid_s2    <= '0';
					pipe_valid_s3    <= '0';
				elsif wb_idx = d - 1 then
					wb_actived <= '1';
				else
					wb_idx <= wb_idx + 1;
				end if;
			else
				rst_counter      <= '1';
				update_completed <= '0';
				done             <= '0';
				bram_valid       <= '0';
				pipe_valid_s1    <= '0';
				pipe_valid_s2    <= '0';
				pipe_valid_s3    <= '0';
			end if;
		end if;
	end process;

	count : popCount
		generic map (20)
		port map (
			clk  => clk,
			rst  => rst_counter,
			en   => counter_en,
			dout => ID
		);

	id_at_last <= '1' when unsigned(ID) = to_unsigned(d - 1, ID'length) else '0';
	counter_en <= run and not id_at_last;

	read_addra <= std_logic_vector(
		to_unsigned(correct_label * d + to_integer(unsigned(ID)), addr_w)
	);
	read_addrb <= std_logic_vector(
		to_unsigned(predicted_label * d + to_integer(unsigned(ID)), addr_w)
	);

	learn_addra <= write_addra when wb_active = '1' else read_addra;
	learn_addrb <= write_addrb when wb_active = '1' else read_addrb;

	learn_mem : blk_mem_gen_LEARN
		port map (
			clka  => clk,
			wea   => learn_wea,
			addra => learn_addra,
			dina  => learn_dina,
			douta => learn_douta,
			clkb  => clk,
			web   => learn_web,
			addrb => learn_addrb,
			dinb  => learn_dinb,
			doutb => learn_doutb
		);
end architecture;
